classdef AGV_Environment < handle
    % CLASSE 2: AGV_Environment
    % Ponte intelligente tra sistema IDRR e agente MAPPO per gestione 
    % traduzione bidirezionale tra rappresentazione interna e formato RL
    
    properties (Access = public)
        % === SISTEMA PRINCIPALE ===
        agvSystem                   % Handle a AGV_System_IDRR_RL
        mappoAgent                  % Handle a MAPPO_Agent (quando disponibile)
        
        % === NODE EMBEDDINGS ===
        nodeEmbeddings              % Matrix 8×N_nodes con embeddings precomputati
        embeddingDim = 8            % Dimensione embeddings
        
        % === MODALITÀ OPERATIVA ===
        operatingMode = 'IDRR'      % 'IDRR' | 'RL_TRAINING' | 'RL_TESTING'
        enableShuffling = true      % Shuffling per prevenire bias posizionale
        
        % === FEATURE DIMENSIONS ===
        globalFeatureDim = 109      % Dimensioni features globali per critic
        localFeatureDim = 109        % Dimensioni features locali per actor
        maxAGVs = 7                 % Numero massimo AGV supportato
        maxTasks = 5                % Numero massimo task nella coda top-5
        
        % === EXPERIENCE BUFFER ===
        openExperiences             % Map agvId -> esperienza aperta
        completedExperiencesByAGV   % Cell array: buffer per ogni AGV
        maxBufferSizePerAGV = 2000  % Dimensione massima buffer per AGV
        totalExperienceCount = 0    % Contatore totale esperienze
        
        % === REWARD PARAMETERS === --> DA TARARE
        lambda = 0.7                % Bilanciamento global vs local reward
        alpha = 0.3                 % Peso waiting time
        beta = 0.2                  % Peso conflict resolution time
        gamma = 0.4                 % Peso travel distance  
        delta = 0.1                 % Peso load imbalance

        % === REWARD NORMALIZATION ===
        sigmoidSteepness = 0.5      % Parametro k per sigmoide throughput
        enableNormalization = true  % Flag normalizzazione reward

        % === THROUGHPUT NORMALIZATION ===
        targetThroughput = 65.0     % [task/h] Target per normalizzazione throughput
        maxThroughputClip = 2.0     % Clip superiore per throughputNorm
                
        % === LOGGING E DEBUG ===
        verboseLogging = false      % Flag logging dettagliato
        debugMode = false           % Flag debug mode
        
        % === VALIDATION COUNTERS ===
        totalDecisionRequests = 0   % Contatore decisioni richieste
        totalExperiences = 0        % Contatore esperienze complete
        positionMappingErrors = 0   % Errori mapping posizione-azione
    end
    
    methods
        function obj = AGV_Environment(agvSystem, varargin)
            % Costruttore AGV_Environment
            
            % Parse input arguments
            p = inputParser;
            addRequired(p, 'agvSystem');
            addParameter(p, 'operatingMode', 'IDRR', @ischar);
            addParameter(p, 'enableShuffling', true, @islogical);
            addParameter(p, 'verboseLogging', false, @islogical);
            addParameter(p, 'debugMode', false, @islogical);
            addParameter(p, 'lambda', 0.7, @isnumeric);
            addParameter(p, 'sigmoidSteepness', 0.5, @isnumeric);  % AGGIUNTO
            parse(p, agvSystem, varargin{:});
            
            % Assegna proprietà
            obj.agvSystem = agvSystem;
            obj.operatingMode = p.Results.operatingMode;
            obj.enableShuffling = p.Results.enableShuffling;
            obj.verboseLogging = p.Results.verboseLogging;
            obj.debugMode = p.Results.debugMode;
            obj.lambda = p.Results.lambda;
            obj.sigmoidSteepness = p.Results.sigmoidSteepness;

            
            % Inizializza componenti
            obj.loadNodeEmbeddings();
            obj.initializeExperienceBuffers();
            
            obj.logMessage('AGV Environment inizializzato');
            obj.logMessage(sprintf('Modalità: %s, Shuffling: %s', ...
                obj.operatingMode, string(obj.enableShuffling)));
        end
        
        function loadNodeEmbeddings(obj)
            % Carica embeddings nodi da file Embeddings.mat
            
            try
                % Carica file embeddings
                embeddingData = load('Embedding.mat');
                
                if isfield(embeddingData, 'Embedded_Nodes')
                    obj.nodeEmbeddings = embeddingData.Embedded_Nodes;
                    [numNodes, dim] = size(obj.nodeEmbeddings);
                    
                    % Verifica dimensioni
                    if dim ~= obj.embeddingDim
                        warning('Dimensione embedding attesa: %d, trovata: %d', ...
                            obj.embeddingDim, dim);
                        obj.embeddingDim = dim;
                    end
                    
                    obj.logMessage(sprintf('✅ Embeddings caricati: %d nodi × %d dim', ...
                        numNodes, obj.embeddingDim));
                else
                    error('Campo Embedded_Nodes non trovato nel file Embeddings.mat');
                end
                
            catch ME
                error('Errore caricamento embeddings: %s', ME.message);
            end
        end

        function initializeExperienceBuffers(obj)
            % Inizializza buffer esperienze separati per ogni AGV

            obj.completedExperiencesByAGV = cell(obj.maxAGVs, 1);
            for i = 1:obj.maxAGVs
                obj.completedExperiencesByAGV{i} = [];
            end

            obj.openExperiences = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            obj.totalExperienceCount = 0;

            obj.logMessage('Experience buffers per AGV inizializzati');
        end

        % === METODI CALCOLO REWARD ===
        
        function rewardComponents = computeRewardComponents(obj, agvId, experience, completedTask)
            % Calcola tutte le componenti raw necessarie per reward

            agv = obj.agvSystem.AGVs(agvId);
            rewardComponents = struct();

            % === THROUGHPUT DERIVATIVE ===
            rewardComponents.throughputDerivative = obj.agvSystem.getThroughputDerivative();

            % === WAITING TIME (solo stato waiting, da currentTaskStateTimes) ===
            if isfield(agv, 'currentTaskStateTimes') && ~isempty(agv.currentTaskStateTimes)
                rewardComponents.waitingTime = agv.currentTaskStateTimes.Waiting;
            else
                rewardComponents.waitingTime = 0;
            end

            % === CONFLICT RESOLUTION TIME (stato resolving) ===
            if isfield(agv, 'currentTaskStateTimes') && ~isempty(agv.currentTaskStateTimes)
                rewardComponents.conflictTime = agv.currentTaskStateTimes.Resolving;
            else
                rewardComponents.conflictTime = 0;
            end

            % === TRAVEL DISTANCE ===
            if isfield(agv, 'taskDistance') && ~isempty(agv.taskDistance)
                rewardComponents.travelDistance = agv.taskDistance;
            elseif isfield(completedTask, 'totalDistance')
                rewardComponents.travelDistance = completedTask.totalDistance;
            else
                rewardComponents.travelDistance = 0;
            end

            % === THEORETICAL MIN TIME ===
            % Calcola distanza minima teorica del task (per normalizzazione WT/CRT)
            taskSel = experience.taskSelected;
            pathToPickup = obj.agvSystem.getOptimalPath(experience.agvPositionAtDecision, taskSel.pickup);
            pathPickupToDropoff = obj.agvSystem.getOptimalPath(taskSel.pickup, taskSel.dropoff);

            distToPickup = obj.agvSystem.calculatePathDistance(pathToPickup);
            distPickupToDropoff = obj.agvSystem.calculatePathDistance(pathPickupToDropoff);
            rewardComponents.theoreticalMinTime = distToPickup + distPickupToDropoff;

            % === AVAILABLE TASK DISTANCES ===
            % Calcola distanze di tutti i task disponibili al momento decisione
            if isfield(experience, 'top5QueueSnapshot') && ~isempty(experience.top5QueueSnapshot)
                queue = experience.top5QueueSnapshot;
                availableDistances = zeros(1, length(queue));

                for i = 1:length(queue)
                    task = queue(i);
                    pathToP = obj.agvSystem.getOptimalPath(experience.agvPositionAtDecision, task.pickup);
                    pathPtoD = obj.agvSystem.getOptimalPath(task.pickup, task.dropoff);
                    availableDistances(i) = obj.agvSystem.calculatePathDistance(pathToP) + ...
                        obj.agvSystem.calculatePathDistance(pathPtoD);
                end
                rewardComponents.availableTaskDistances = availableDistances;
            else
                rewardComponents.availableTaskDistances = [];
            end

            % === LOAD IMBALANCE ===
            totalTasks = sum([obj.agvSystem.AGVs.tasksCompleted]);
            avgTasks = totalTasks / obj.agvSystem.nAGV;
            rewardComponents.loadImbalance = abs(agv.tasksCompleted - avgTasks);
            rewardComponents.totalSystemTasks = totalTasks;
        end

        function [totalReward, globalReward, localReward] = computeNormalizedReward(obj, components)
            % Calcola reward normalizzate da componenti raw

            if ~obj.enableNormalization
                % Fallback: reward non normalizzate
                globalReward = components.throughputDerivative;
                localReward = -obj.alpha * components.waitingTime ...
                    -obj.beta * components.conflictTime ...
                    -obj.gamma * components.travelDistance ...
                    -obj.delta * components.loadImbalance;
                totalReward = obj.lambda * globalReward + (1 - obj.lambda) * localReward;
                return;
            end

            % === NORMALIZZAZIONE COMPONENTI ===

            % 1. Throughput Derivative: sigmoide [0,1]
            tpNorm = obj.normalizeThroughput(components.throughputDerivative);

            % 2. Waiting Time: frazione rispetto tempo teorico
            wtNorm = obj.normalizeWaitingTime(components.waitingTime, components.theoreticalMinTime);

            % 3. Conflict Resolution Time: frazione rispetto tempo teorico
            crtNorm = obj.normalizeConflictTime(components.conflictTime, components.theoreticalMinTime);

            % 4. Travel Distance: rispetto alternative disponibili
            tdNorm = obj.normalizeTravelDistance(components.travelDistance, ...
                components.availableTaskDistances);

            % 5. Load Imbalance: rispetto stato sistema
            liNorm = obj.normalizeLoadImbalance(components.loadImbalance, ...
                components.totalSystemTasks);

            % === CONVERSIONE IN BENEFICI ===

            % Throughput: [0,1] → [0,2], dove 1.0 = stabile
            benefitTP = 2.0 * tpNorm;

            % Waiting Time: 1/(1+norm), valori alti norm = benefit basso
            benefitWT = 1.0 / (1.0 + wtNorm);

            % Conflict Time: 1/(1+norm)
            benefitCRT = 1.0 / (1.0 + crtNorm);

            % Travel Distance: 1-norm, distanza bassa = benefit alto
            benefitTD = 1.0 - tdNorm;

            % Load Imbalance: 1-norm, sbilanciamento basso = benefit alto
            benefitLI = 1.0 - liNorm;

            % === REWARD FINALE ===

            localReward = obj.alpha * benefitWT + obj.beta * benefitCRT + ...
                obj.gamma * benefitTD + obj.delta * benefitLI;

            globalReward = benefitTP;

            rawReward = obj.lambda * globalReward + (1 - obj.lambda) * localReward;

            % Normalizzazione finale in [0,1]
            maxPossibleReward = obj.lambda * 2.0 + (1 - obj.lambda) * 1.0;
            totalReward = rawReward / maxPossibleReward;
            totalReward = max(0, min(1, totalReward));

            % Debug logging
            if obj.debugMode
                fprintf('    WT: %.2fs / %.2fs = %.3f → benefit: %.3f\n', ...
                    components.waitingTime, components.theoreticalMinTime, wtNorm, benefitWT);
                fprintf('    CRT: %.2fs / %.2fs = %.3f → benefit: %.3f\n', ...
                    components.conflictTime, components.theoreticalMinTime, crtNorm, benefitCRT);
                fprintf('    TD: %.1fm, norm: %.3f → benefit: %.3f\n', ...
                    components.travelDistance, tdNorm, benefitTD);
                fprintf('    LI: %.2f, norm: %.3f → benefit: %.3f\n', ...
                    components.loadImbalance, liNorm, benefitLI);
                fprintf('    TP deriv: %+.4f → sigmoid: %.3f → benefit: %.3f\n', ...
                    components.throughputDerivative, tpNorm, benefitTP);
                fprintf('    → Total Reward: %.4f\n', totalReward);
            end
        end

        function tpNorm = normalizeThroughput(obj, tpDeriv)
            % Sigmoide: mappa [-inf,+inf] → [0,1], con 0→0.5
            k = obj.sigmoidSteepness;
            tpNorm = 1.0 / (1.0 + exp(-k * tpDeriv));
            tpNorm = max(0, min(1, tpNorm));
        end

        function wtNorm = normalizeWaitingTime(obj, wt, theoreticalMinTime)
            % Frazione tempo extra rispetto al minimo teorico
            if theoreticalMinTime > 0
                wtNorm = wt / theoreticalMinTime;
            else
                wtNorm = 0;
            end
            wtNorm = max(0, wtNorm);
        end

        function crtNorm = normalizeConflictTime(obj, crt, theoreticalMinTime)
            % Frazione tempo extra rispetto al minimo teorico
            if theoreticalMinTime > 0
                crtNorm = crt / theoreticalMinTime;
            else
                crtNorm = 0;
            end
            crtNorm = max(0, crtNorm);
        end

        function tdNorm = normalizeTravelDistance(obj, td, availableDistances)
            % Normalizza rispetto min/max alternative disponibili
            if isempty(availableDistances) || length(availableDistances) < 2
                tdNorm = 0;
            else
                minDist = min(availableDistances);
                maxDist = max(availableDistances);
                if maxDist > minDist
                    tdNorm = (td - minDist) / (maxDist - minDist);
                else
                    tdNorm = 0;
                end
            end
            tdNorm = max(0, min(1, tdNorm));
        end

        function liNorm = normalizeLoadImbalance(obj, li, totalSystemTasks)
            if totalSystemTasks > 0
                % Massimo realistico: deviazione standard teorica
                avgTasks = totalSystemTasks / obj.agvSystem.nAGV;
                maxRealistic = avgTasks;  % Normalizza rispetto alla media
                liNorm = li / maxRealistic;
            else
                liNorm = 0;
            end
            liNorm = max(0, min(1, liNorm));
        end

        % === INTERFACCIA PRINCIPALE CON CLASSE 1 ===
        
        function handleDecisionRequest(obj, agvId, agvSystem)
            % Gestisce richiesta di decisione da AGV idle
            
            obj.totalDecisionRequests = obj.totalDecisionRequests + 1;
            
            try
                % Ottieni coda top-5 dal sistema
                top5Queue = agvSystem.getTop5TaskQueue();
                
                if isempty(top5Queue)
                    obj.logMessage(sprintf('⚠️ Coda top-5 vuota per AGV %d - fallback IDRR', agvId));
                    agvSystem.assignTaskIDRR(agvId);
                    return;
                end
                
                % Applica shuffling se in training
                [shuffledQueue, shuffleMapping] = obj.applyShuffling(top5Queue);
                
                % Calcola features
                [localFeatures, globalFeatures] = obj.computeFeatures(agvId, shuffledQueue);

                % Crea action Mask
                actionMask = obj.createActionMask(shuffledQueue);
                
                % Avvia esperienza
                obj.startExperience(agvId, localFeatures, globalFeatures, shuffledQueue, shuffleMapping);
                
                % Richiedi decisione all'agente
                if ~isempty(obj.mappoAgent)
                    try
                        if strcmp (obj.operatingMode,'RL_TESTING')
                            selectedPosition = obj.mappoAgent.selectActionDeterministic(localFeatures, globalFeatures, actionMask);
                        else
                            selectedPosition = obj.mappoAgent.selectAction(localFeatures, globalFeatures, actionMask);
                        end

                        % Valida azione
                        if obj.validateAction(selectedPosition, length(shuffledQueue))
                            % Mappa posizione a task originale
                            originalPosition = obj.mapPositionToOriginal(selectedPosition, shuffleMapping);
                            selectedTask = top5Queue(originalPosition);

                            % Completa parte decisionale dell'esperienza
                            obj.recordDecision(agvId, selectedPosition, selectedTask, originalPosition);

                            % Esegui decisione nel sistema
                            success = agvSystem.executeRLDecision(agvId, originalPosition);

                            if ~success
                                obj.logMessage(sprintf('❌ Esecuzione fallita per AGV %d - fallback IDRR', agvId));
                                obj.cancelExperience(agvId);
                                agvSystem.assignTaskIDRR(agvId);
                            end
                        else
                            obj.positionMappingErrors = obj.positionMappingErrors + 1;
                            obj.logMessage(sprintf('❌ Azione non valida da agente: %d', selectedPosition));
                            obj.cancelExperience(agvId);
                            agvSystem.assignTaskIDRR(agvId);
                        end

                    catch ME
                        obj.logMessage(sprintf('❌ Errore agente MAPPO: %s', ME.message));
                        obj.cancelExperience(agvId);
                        agvSystem.assignTaskIDRR(agvId);
                    end
                else
                    % Nessun agente disponibile - fallback IDRR
                    obj.cancelExperience(agvId);
                    agvSystem.assignTaskIDRR(agvId);
                end
                
            catch ME
                obj.logMessage(sprintf('❌ Errore gestione decisione AGV %d: %s', agvId, ME.message));
                agvSystem.assignTaskIDRR(agvId);
            end
        end
        
        function handleTaskCompletion(obj, agvId, completedTask, isDone)
            % Notifica completamento task per finalizzare esperienza
            
            try
                if obj.openExperiences.isKey(agvId)
                    % Calcola osservazioni next
                    currentTop5 = obj.agvSystem.getTop5TaskQueue();
                    [nextLocalFeatures, nextGlobalFeatures] = obj.computeFeatures(agvId, currentTop5);
                    
                    % Completa esperienza
                    obj.completeExperience(agvId, nextLocalFeatures, nextGlobalFeatures, ...
                        completedTask, isDone);
                    
                    obj.totalExperiences = obj.totalExperiences + 1;
                    obj.logMessage(sprintf('✅ Esperienza completa per AGV %d', agvId));
                    
                else
                    % AGV completato task senza esperienza RL (normale in modalità IDRR)
                    if obj.debugMode
                        obj.logMessage(sprintf('📝 Task completato AGV %d senza esperienza RL', agvId));
                    end
                end
                
            catch ME
                obj.logMessage(sprintf('❌ Errore completamento esperienza AGV %d: %s', agvId, ME.message));
            end
        end

        function notifyEpisodeComplete(obj, isRunOut)
            % isRunOut: true se episodio terminato per timeout, false se completato normalmente

            if nargin < 2
                isRunOut = false;
            end

            if isRunOut
                obj.logMessage('⏰ === EPISODIO TERMINATO PER RUN OUT (TIMEOUT) ===');
                % Chiudi tutte le esperienze aperte con penalità
                obj.closeAllOpenExperiencesWithPenalty();
            else
                obj.logMessage('🎯 === EPISODIO COMPLETATO NORMALMENTE ===');
            end

            totalExp = obj.getTotalExperienceCount();
            obj.logMessage(sprintf('📦 Esperienze raccolte: %d totali', totalExp));

            for agvId = 1:obj.maxAGVs
                numExp = length(obj.completedExperiencesByAGV{agvId});
                if numExp > 0
                    obj.logMessage(sprintf('   AGV_%d: %d esperienze', agvId, numExp));
                end
            end

            % Trasferisci esperienze SOLO in modalità TRAINING
            if ~isempty(obj.mappoAgent) && strcmp(obj.operatingMode, 'RL_TRAINING')
                try
                    obj.logMessage('🔄 Trasferimento esperienze → MAPPO_Agent...');
                    numTransferred = obj.transferExperiencesToAgent();
                    obj.logMessage(sprintf('✅ Trasferite %d esperienze', numTransferred));
                catch ME
                    obj.logMessage(sprintf('❌ Errore trasferimento: %s', ME.message));
                end
            end
        end

        function closeAllOpenExperiencesWithPenalty(obj)
            % Chiude tutte le esperienze aperte con penalità per run out

            if obj.openExperiences.Count == 0
                obj.logMessage('   Nessuna esperienza aperta da chiudere');
                return;
            end

            obj.logMessage(sprintf('   🚨 Chiusura %d esperienze aperte con penalità...', obj.openExperiences.Count));

            % Ottieni tutti gli AGV con esperienze aperte
            openAGVs = cell2mat(obj.openExperiences.keys);

            for agvId = openAGVs
                try
                    experience = obj.openExperiences(agvId);

                    % Calcola osservazioni next (stato attuale)
                    currentTop5 = obj.agvSystem.getTop5TaskQueue();
                    [nextLocalFeatures, nextGlobalFeatures] = obj.computeFeatures(agvId, currentTop5);

                    % CALCOLA PENALITÀ PER RUN OUT
                    % La penalità deve essere informativa ma non eccessiva
                    runOutPenalty = obj.computeRunOutPenalty(agvId, experience);

                    % Completa esperienza
                    experience.nextLocalFeatures = nextLocalFeatures;
                    experience.nextGlobalFeatures = nextGlobalFeatures;
                    experience.nextTop5Queue = currentTop5;

                    % Assegna reward penalizzato
                    experience.reward = runOutPenalty;
                    experience.globalReward = runOutPenalty * obj.lambda;
                    experience.localReward = runOutPenalty * (1 - obj.lambda);

                    % Flag specifico per run out
                    experience.timestampCompletion = obj.agvSystem.currentTime;
                    experience.completedTask = struct(); % Task NON completato
                    experience.taskCompleted = false;
                    experience.done = true; % Episodio terminato
                    experience.isRunOut = true; % Flag specifico

                    % Sposta in buffer completate
                    obj.completedExperiencesByAGV{agvId} = ...
                        [obj.completedExperiencesByAGV{agvId}, experience];
                    obj.totalExperienceCount = obj.totalExperienceCount + 1;

                    obj.logMessage(sprintf('      AGV_%d: esperienza chiusa con penalità = %.4f', ...
                        agvId, runOutPenalty));

                catch ME
                    obj.logMessage(sprintf('      ❌ Errore chiusura AGV_%d: %s', agvId, ME.message));
                end
            end

            % Svuota tutte le esperienze aperte
            obj.openExperiences = containers.Map('KeyType', 'int32', 'ValueType', 'any');

            obj.logMessage('   ✅ Tutte le esperienze aperte chiuse');
        end

        function penalty = computeRunOutPenalty(obj, agvId, experience)
            % Calcola penalità proporzionata e informativa per run out
            %
            % PRINCIPI:
            % 1. Penalità base per episodio non completato
            % 2. Penalità modulata per progresso task
            % 3. Penalità aggiuntiva per stati inefficienti (waiting/resolving)
            % 4. Penalità proporzionale alla durata degli stati inefficienti
            % 5. Scala coerente con reward normali per non confondere l'agente

            % ========== PARAMETRI PENALITÀ (CONFIGURABILI) ==========
            BASE_PENALTY = -0.5;           % Penalità base per run out
            PROGRESS_FACTOR = 0.3;         % Modulazione progresso task

            % Penalità per stati inefficienti (per secondo)
            WAITING_PENALTY_RATE = -0.015;    % Penalità per secondo in waiting
            RESOLVING_PENALTY_RATE = -0.025;  % Penalità per secondo in resolving (più grave)

            % Limiti penalità stati inefficienti
            MAX_STATE_PENALTY = -0.4;      % Cap massimo penalità stati
            STATE_TIME_THRESHOLD = 10.0;    % Sotto 5s, penalità ridotta (transitori)

            % Penalità totale
            totalPenalty = BASE_PENALTY;

            % ========== COMPONENTE 1: PENALITÀ PROGRESSO TASK ==========
            taskProgress = 0.5; % Default: task a metà

            if ~isempty(obj.agvSystem.AGVs(agvId).task)
                agv = obj.agvSystem.AGVs(agvId);

                % Stima progresso in base allo stato AGV e percorso residuo
                if agv.state == 0
                    % Idle: nessun progresso (appena assegnato)
                    taskProgress = 0.0;
                elseif agv.state == 1 && ~isempty(agv.residualRoute)
                    % Moving: stima progresso in base a distanza residua
                    totalPlanned = agv.totalPlannedDistance;
                    if totalPlanned > 0
                        travelled = totalPlanned - agv.taskDistance;
                        taskProgress = travelled / totalPlanned;
                        taskProgress = min(1.0, max(0.0, taskProgress));
                    end
                elseif agv.state == 2 || agv.state == 3
                    % Waiting/Resolving: task in corso avanzato
                    taskProgress = 0.7;
                end
            end

            % Modula penalità base per progresso
            progressPenalty = -(PROGRESS_FACTOR * taskProgress);
            totalPenalty = totalPenalty + progressPenalty;

            % ========== COMPONENTE 2: PENALITÀ STATI INEFFICIENTI ==========
            agv = obj.agvSystem.AGVs(agvId);
            statePenalty = 0;
            stateInfo = '';

            % Verifica stato corrente AGV
            currentState = agv.state;

            if currentState == 2  % WAITING
                % Calcola tempo in waiting
                waitingTime = agv.waitingTime; % Tempo accumulato in waiting

                if waitingTime > STATE_TIME_THRESHOLD
                    % Penalità proporzionale al tempo in waiting
                    % Solo se supera soglia (evita penalizzare transitori)
                    effectiveTime = waitingTime - STATE_TIME_THRESHOLD;
                    statePenalty = WAITING_PENALTY_RATE * effectiveTime;
                    statePenalty = max(MAX_STATE_PENALTY, statePenalty); % Cap

                    stateInfo = sprintf('waiting=%.1fs', waitingTime);
                end

            elseif currentState == 3  % RESOLVING
                % Per resolving, stima tempo in questo stato
                % Usa il tempo dal task start come proxy
                if agv.currentTaskStartTime > 0
                    timeInTask = obj.agvSystem.currentTime - agv.currentTaskStartTime;

                    % Assumi che se è in resolving e ha speso molto tempo,
                    % ha passato tempo significativo risolvendo conflitti
                    if timeInTask > STATE_TIME_THRESHOLD
                        effectiveTime = timeInTask - STATE_TIME_THRESHOLD;
                        statePenalty = RESOLVING_PENALTY_RATE * effectiveTime;
                        statePenalty = max(MAX_STATE_PENALTY, statePenalty);

                        stateInfo = sprintf('resolving≈%.1fs', timeInTask);
                    end
                end

                % ALTERNATIVA: Usa conflictsCumulative se disponibile
                % Se l'AGV ha generato molti conflitti, penalizza ulteriormente
                if agv.conflictsCumulative > 0
                    conflictPenalty = -0.05 * agv.conflictsCumulative;
                    conflictPenalty = max(-0.2, conflictPenalty); % Cap a -0.2
                    statePenalty = statePenalty + conflictPenalty;

                    stateInfo = sprintf('%s, conflicts=%d', stateInfo, agv.conflictsCumulative);
                end
            end

            % Aggiungi penalità stati alla penalità totale
            totalPenalty = totalPenalty + statePenalty;

            % ========== CLAMP FINALE ==========
            % Assicura che penalità totale rimanga in range ragionevole
            % Range: [-1.5, -0.3] - più ampio per catturare casi molto negativi
            totalPenalty = max(-1.5, min(-0.3, totalPenalty));

            % ========== LOGGING DETTAGLIATO ==========
            if obj.verboseLogging
                obj.logMessage(sprintf('      AGV_%d Penalità Run Out:', agvId));
                obj.logMessage(sprintf('        - Base: %.3f', BASE_PENALTY));
                obj.logMessage(sprintf('        - Progresso task: %.3f (progress=%.2f)', ...
                    progressPenalty, taskProgress));
                if statePenalty < 0
                    obj.logMessage(sprintf('        - Stato inefficiente: %.3f (%s)', ...
                        statePenalty, stateInfo));
                end
                obj.logMessage(sprintf('        → Penalità TOTALE: %.3f', totalPenalty));
            end

            penalty = totalPenalty;
        end
        % === COMPUTAZIONE FEATURES ===
        
        function [localFeatures, globalFeatures] = computeFeatures(obj, requestingAGVId, top5Queue)
            % Calcola features globali per l'AGV richiedente
            globalFeatures = obj.computeGlobalFeatures(requestingAGVId, top5Queue);
            % Calcola features locali specifiche per AGV
            localFeatures = obj.computeLocalFeatures(requestingAGVId, top5Queue);
        end

        function actionMask = createActionMask(obj, top5Queue)
            % Input:
            %   top5Queue - Coda task corrente (length <= 5)
            % Output:
            %   actionMask - [5x1] logical, true per posizioni valide

            numValidActions = length(top5Queue);
            actionMask = false(5, 1);
            actionMask(1:numValidActions) = true;

            if obj.debugMode
                obj.logMessage(sprintf('Action mask: %d/%d azioni valide', ...
                    numValidActions, 5));
            end
        end
        
        function globalFeatures = computeGlobalFeatures(obj, requestingAGVId, top5Queue)
            % Computa features globali (241 dimensioni) per il critic

            features = [];

            % === AGV-RELATED FEATURES ===

            agv = obj.agvSystem.AGVs(requestingAGVId);

            % AGV Position Embedding (8 dim)
            posEmb = obj.getNodeEmbedding(agv.logicalNode);
   
            % Combina features AGV
            agvFeatures = posEmb';

            features = [features; agvFeatures];

            % === SYSTEM-RELATED FEATURES ===

            % Global Congestion Embedding (8 dim)
            congestionEmb = obj.computeGlobalCongestionEmbedding();

            % Top-5 Available Tasks (5 dim) - ID task
            taskIDs = obj.encodeTaskIDs(top5Queue);

            % Task Path Embeddings (40 dim = 5×8) - DA POSIZIONE AGV CANDIDATO
            taskPathEmbs = obj.computeTaskPathEmbeddingsFromAGV(requestingAGVId, top5Queue);

            % Task Distances (5 dim) - DA POSIZIONE AGV CANDIDATO
            taskDistances = obj.computeTaskDistancesFromAGV(requestingAGVId, top5Queue);

            % Future Congestion Embeddings (40 dim = 5×8) - PER AGV CANDIDATO
            futureCongestionEmbs = obj.computePossibleCongestionEmbeddings(requestingAGVId, top5Queue);

            % System Metrics (3 dim)
            systemMetrics = obj.computeSystemMetrics();

            % Combina system features
            systemFeatures = [congestionEmb; taskIDs'; taskPathEmbs(:); taskDistances'; ...
                futureCongestionEmbs(:); systemMetrics'];

            % Features globali complete
            globalFeatures = [features; systemFeatures];

            % Verifica dimensioni
            if length(globalFeatures) ~= obj.globalFeatureDim
                warning('Dimensioni features globali: attese %d, ottenute %d', ...
                    obj.globalFeatureDim, length(globalFeatures));
            end
        end
            
        function localFeatures = computeLocalFeatures(obj, agvId, top5Queue)
            % Computa features globali (241 dimensioni) per il critic

            features = [];

            % === AGV-RELATED FEATURES ===

            agv = obj.agvSystem.AGVs(agvId);

            % AGV Position Embedding (8 dim)
            posEmb = obj.getNodeEmbedding(agv.logicalNode);

            % Combina features AGV
            agvFeatures = posEmb';

            features = [features; agvFeatures];

            % === SYSTEM-RELATED FEATURES ===

            % Global Congestion Embedding (8 dim)
            congestionEmb = obj.computeGlobalCongestionEmbedding();

            % Top-5 Available Tasks (5 dim) - ID task
            taskIDs = obj.encodeTaskIDs(top5Queue);

            % Task Path Embeddings (40 dim = 5×8) - DA POSIZIONE AGV CANDIDATO
            taskPathEmbs = obj.computeTaskPathEmbeddingsFromAGV(agvId, top5Queue);

            % Task Distances (5 dim) - DA POSIZIONE AGV CANDIDATO
            taskDistances = obj.computeTaskDistancesFromAGV(agvId, top5Queue);

            % Future Congestion Embeddings (40 dim = 5×8) - PER AGV CANDIDATO
            futureCongestionEmbs = obj.computePossibleCongestionEmbeddings(agvId, top5Queue);

            % System Metrics (3 dim)
            systemMetrics = obj.computeSystemMetrics();

            % Combina system features
            systemFeatures = [congestionEmb; taskIDs'; taskPathEmbs(:); taskDistances'; ...
                futureCongestionEmbs(:); systemMetrics'];

            % Features globali complete
            localFeatures = [features; systemFeatures];

            % Verifica dimensioni
            if length(localFeatures) ~= obj.localFeatureDim
                warning('Dimensioni features globali: attese %d, ottenute %d', ...
                    obj.localFeatureDim, length(localFeatures));
            end
        end
        
        % === COMPUTAZIONE EMBEDDINGS PESATI ===
        
        function embedding = computeWeightedPathEmbedding(obj, path, agv)
            % Calcola embedding pesato di un percorso usando formula (11)
            %
            % PARAMETRI:
            %   path - Array di nodi che formano il percorso
            %   agv - (opzionale) Riferimento all'AGV per calcolare offset se in movimento

            if isempty(path)
                embedding = zeros(obj.embeddingDim, 1);
                return;
            end

            % Calcola distanze cumulative reali
            distances = zeros(length(path), 1);

            
            if agv.isMoving && ~isempty(agv.residualRoute) && agv.residualRoute(1) == path(1)
                % AGV sta raggiungendo il primo nodo del percorso
                % Calcola tempo/distanza rimanente (velocità = 1 m/s)
                timeToArrival = agv.arrivalTime - obj.agvSystem.currentTime;
                distances(1) = max(0, timeToArrival); % Evita valori negativi per arrotondamenti
            end
        
            
            for i = 2:length(path)
                % Usa distanze reali dal sistema IDRR
                segmentDist = obj.agvSystem.edges(path(i-1), path(i));
                distances(i) = distances(i-1) + segmentDist;
            end

            % Calcola pesi: wi = 1/(1 + Di)
            weights = 1 ./ (1 + distances);

            % Calcola embedding pesato
            weightedSum = zeros(obj.embeddingDim, 1);
            totalWeight = 0;

            for i = 1:length(path)
                nodeEmb = obj.getNodeEmbedding(path(i));
                weightedSum = weightedSum + weights(i) * nodeEmb';
                totalWeight = totalWeight + weights(i);
            end

            if totalWeight > 0
                embedding = weightedSum / totalWeight;
            else
                embedding = zeros(obj.embeddingDim, 1);
            end
        end
        
        function embedding = computeWeightedOverlapEmbedding(obj, candidatePath, activePaths)
            % Calcola weighted overlap embedding tra percorso candidato e percorsi attivi

            if isempty(candidatePath) || isempty(activePaths)
                embedding = zeros(obj.embeddingDim, 1);
                return;
            end

            overlapEmbeddings = [];
            numValidOverlaps = 0;

            % Per ogni percorso attivo
            for j = 1:length(activePaths)
                activePath = activePaths{j};
                if isempty(activePath), continue; end

                % Trova nodi overlap
                overlapNodes = intersect(candidatePath, activePath);

                if ~isempty(overlapNodes)
                    % Calcola embedding overlap per questo percorso
                    % NOTA: candidateAGV non disponibile qui, solo activeAGV disponibile tramite indice
                    % Recupera riferimento AGV attivo
                    activeAGVRef = [];
                    if j <= obj.agvSystem.nAGV
                        activeAGVRef = obj.agvSystem.AGVs(j);
                    end

                    overlapEmb = obj.computeOverlapEmbeddingForPath(...
                        candidatePath, activePath, overlapNodes, [], activeAGVRef);
                    overlapEmbeddings = [overlapEmbeddings, overlapEmb];
                    numValidOverlaps = numValidOverlaps + 1;
                end
            end

            % Media degli overlap embeddings
            if numValidOverlaps > 0
                embedding = mean(overlapEmbeddings, 2);
            else
                embedding = zeros(obj.embeddingDim, 1);
            end
        end

        function overlapEmb = computeOverlapEmbeddingForPath(obj, candidatePath, activePath, overlapNodes, candidateAGV, activeAGV)
            % Calcola overlap embedding tra due percorsi specifici
            %
            % PARAMETRI:
            %   candidatePath - Percorso del task candidato
            %   activePath - Percorso attivo dell'AGV
            %   overlapNodes - Nodi di overlap tra i due percorsi
            %   candidateAGV - (opzionale) Riferimento AGV candidato
            %   activeAGV - (opzionale) Riferimento AGV attivo

            weightedSum = zeros(obj.embeddingDim, 1);
            totalWeight = 0;

            % Gestione parametri opzionali
            if nargin < 5
                candidateAGV = [];
            end
            if nargin < 6
                activeAGV = [];
            end

            for i = 1:length(overlapNodes)
                node = overlapNodes(i);

                % Distanza da AGV candidato al nodo overlap (con offset se in movimento)
                Dc = obj.computeDistanceToNode(candidatePath, node, candidateAGV);

                % Distanza da AGV attivo al nodo overlap (con offset se in movimento)
                Dj = obj.computeDistanceToNode(activePath, node, activeAGV);

                % Peso: w(Ni) = 1/((1 + Dc)(1 + Dj))
                weight = 1 / ((1 + Dc) * (1 + Dj));

                % Embedding del nodo
                nodeEmb = obj.getNodeEmbedding(node);

                weightedSum = weightedSum + weight * nodeEmb;
                totalWeight = totalWeight + weight;
            end

            if totalWeight > 0
                overlapEmb = weightedSum / totalWeight;
            else
                overlapEmb = zeros(obj.embeddingDim, 1);
            end
        end

        function distance = computeDistanceToNode(obj, path, targetNode, agv)
            % Calcola distanza reale lungo percorso fino a nodo target
            %
            % PARAMETRI:
            %   path - Array di nodi che formano il percorso
            %   targetNode - Nodo target da raggiungere
            %   agv - (opzionale) Riferimento all'AGV per calcolare offset se in movimento

            distance = 0;
            targetIdx = find(path == targetNode, 1);

            if isempty(targetIdx)
                distance = inf;
                return;
            end

            % CASO PARTICOLARE: se AGV in movimento verso primo nodo del percorso
            if nargin >= 4 && ~isempty(agv)
                if agv.isMoving && ~isempty(agv.residualRoute) && agv.residualRoute(1) == path(1)
                    % AGV sta raggiungendo il primo nodo del percorso
                    timeToArrival = agv.arrivalTime - obj.agvSystem.currentTime;
                    distance = max(0, timeToArrival);
                end
            end

            % Somma distanze reali lungo il percorso
            for i = 1:(targetIdx-1)
                segmentDist = obj.agvSystem.edges(path(i), path(i+1));
                distance = distance + segmentDist;
            end
        end
        
        % === COMPUTAZIONE FEATURES SPECIFICHE ===
     
        function residualDist = computeResidualDistance(obj, agv)
            % Calcola distanza residua reale AGV (NORMALIZZATA)
            %
            % NORMALIZZAZIONE:
            % - Se AGV sta eseguendo un task: normalizza rispetto alla distanza totale del task
            % - Altrimenti: normalizza rispetto alla massima distanza globale possibile
            %
            % Questo permette alla rete di interpretare la residual distance come
            % "percentuale di completamento" del task corrente.

            if isempty(agv.residualRoute)
                residualDist = 0;
                return;
            end

            % Calcola distanza residua raw
            rawResidualDist = 0;

            % CASO PARTICOLARE: se AGV in movimento verso primo nodo
            if agv.isMoving
                timeToArrival = agv.arrivalTime - obj.agvSystem.currentTime;
                rawResidualDist = max(0, timeToArrival);
            end

            % Somma distanze lungo percorso residuo
            for i = 1:(length(agv.residualRoute)-1)
                segmentDist = obj.agvSystem.edges(...
                    agv.residualRoute(i), agv.residualRoute(i+1));
                rawResidualDist = rawResidualDist + segmentDist;
            end

            % NORMALIZZAZIONE

            % AGV sta eseguendo un task con distanza pianificata nota
            % Normalizza rispetto alla distanza totale del task specifico
            % Questo fornisce una "percentuale di completamento" del task
            plannedDist = agv.totalPlannedDistance;
            
            residualDist = rawResidualDist / plannedDist;

            if obj.debugMode && mod(obj.agvSystem.currentTime, 10) < obj.agvSystem.timeStep
                completionPercentage = (1 - residualDist) * 100;
                fprintf('   AGV %d: residual dist = %.2f (%.1f%% completed)\n', ...
                    agv.id, residualDist, completionPercentage);
            end

        end

        function congestionEmb = computeGlobalCongestionEmbedding(obj)
            % Calcola embedding congestione globale sistema

            totalWeight = 0;
            weightedSum = zeros(obj.embeddingDim, 1);

            % Raccogli percorsi attivi di tutti gli AGV
            for agvIdx = 1:obj.agvSystem.nAGV
                agv = obj.agvSystem.AGVs(agvIdx);
                if ~isempty(agv.residualRoute)
                    % Passa riferimento AGV per gestire caso in movimento
                    pathEmb = obj.computeWeightedPathEmbedding(agv.residualRoute, agv);
                    residualDist = obj.computeResidualDistance(agv);

                    % Peso: αj = 1/(1 + L^(j))
                    weight = 1 / (1 + residualDist);

                    weightedSum = weightedSum + weight * pathEmb;
                    totalWeight = totalWeight + weight;
                end
            end

            if totalWeight > 0
                congestionEmb = weightedSum / totalWeight;
            else
                congestionEmb = zeros(obj.embeddingDim, 1);
            end
        end
        
        function taskIDs = encodeTaskIDs(obj, top5Queue)
            % Codifica ID task nella coda top-5
            
            taskIDs = zeros(1, obj.maxTasks);
            
            for i = 1:min(length(top5Queue), obj.maxTasks)
                if ~isempty(top5Queue(i).taskTypeId)
                    taskIDs(i) = double(top5Queue(i).taskTypeId)/14; % ID univoco 1-14
                end
            end
        end
        
        function pathEmbs = computeTaskPathEmbeddingsFromAGV(obj, agvId, top5Queue)
            % Calcola embeddings percorsi task da posizione specifica AGV

            pathEmbs = zeros(obj.embeddingDim, obj.maxTasks);
            agv = obj.agvSystem.AGVs(agvId);

            for i = 1:min(length(top5Queue), obj.maxTasks)
                task = top5Queue(i);
                % AGV fermo: percorso completo da posizione corrente
                toPickup = obj.agvSystem.getOptimalPath(agv.logicalNode, task.pickup);
                toDropoff = obj.agvSystem.getOptimalPath(task.pickup, task.dropoff);
                fullPath = [toPickup, toDropoff(2:end)];
                pathEmbs(:, i) = obj.computeWeightedPathEmbedding(fullPath,obj.agvSystem.AGVs(agvId));
            end

        end

        function distances = computeTaskDistancesFromAGV(obj, agvId, top5Queue)
            % Calcola distanze task da posizione specifica AGV
            % NORMALIZZATE rispetto alla massima distanza nel pool dei 5 task candidati
            %
            % Questa normalizzazione permette alla rete neurale di confrontare i task
            % nel contesto specifico della decisione corrente, facilitando l'apprendimento
            % dell'obiettivo di minimizzazione della distanza percorsa.

            distances = zeros(1, obj.maxTasks);
            agv = obj.agvSystem.AGVs(agvId);

            % Array temporaneo per le distanze raw
            rawDistances = zeros(1, obj.maxTasks);

            % Calcola distanze raw per tutti i task nel pool
            for i = 1:min(length(top5Queue), obj.maxTasks)
                task = top5Queue(i);

                % Distanza totale da posizione corrente AGV al completamento task
                PathToPickup = obj.agvSystem.getOptimalPath(agv.logicalNode, task.pickup);
                toPickupDist = obj.agvSystem.calculatePathDistance(PathToPickup);

                PathToDropoff = obj.agvSystem.getOptimalPath(task.pickup, task.dropoff);
                pickupToDropoffDist = obj.agvSystem.calculatePathDistance(PathToDropoff);

                rawDistances(i) = toPickupDist + pickupToDropoffDist;
            end

            % NORMALIZZAZIONE RELATIVA AL POOL DEI 5 TASK
            numValidTasks = min(length(top5Queue), obj.maxTasks);

            if numValidTasks > 0
                validRawDistances = rawDistances(1:numValidTasks);
                maxPoolDistance = max(validRawDistances);

                if maxPoolDistance > 0
                    % Normalizza rispetto al massimo nel pool corrente
                    % Task più lontano → 1.0, task più vicino → valore < 1.0
                    distances(1:numValidTasks) = validRawDistances / maxPoolDistance;
                else
                    % Caso edge: tutte distanze zero (AGV già in posizione dei task)
                    % Imposta tutte a 0 per indicare equivalenza
                    distances(1:numValidTasks) = 0;
                end

                if obj.debugMode
                    fprintf('   📐 Task distances (normalized): [%.2f, %.2f, %.2f, %.2f, %.2f]\n', ...
                        distances(1), distances(2), distances(3), distances(4), distances(5));
                    fprintf('      Raw distances: [%.1f, %.1f, %.1f, %.1f, %.1f] m\n', ...
                        rawDistances(1), rawDistances(2), rawDistances(3), rawDistances(4), rawDistances(5));
                    fprintf('      Max pool distance: %.1f m\n', maxPoolDistance);
                end
            end

            % Task non validi rimangono a 0 (padding)
            % Questo è coerente perché action masking li escluderà comunque
        end
        
        function congestionEmbs = computePossibleCongestionEmbeddings(obj, agvId, top5Queue)
            % Calcola embeddings congestione possibile per AGV specifico

            congestionEmbs = zeros(obj.embeddingDim, obj.maxTasks);
            agv = obj.agvSystem.AGVs(agvId);

            % Raccogli percorsi attivi degli ALTRI AGV
            activePaths = {};
            activeAGVIndices = [];

            for otherAGVIdx = 1:obj.agvSystem.nAGV
                if otherAGVIdx ~= agvId
                    otherAGV = obj.agvSystem.AGVs(otherAGVIdx);
                    if ~isempty(otherAGV.residualRoute)
                        activePaths{end+1} = otherAGV.residualRoute;
                        activeAGVIndices(end+1) = otherAGVIdx;
                    end
                end
            end

            for i = 1:min(length(top5Queue), obj.maxTasks)
                task = top5Queue(i);

                % Percorso che questo AGV seguirebbe per il task
                toPickup = obj.agvSystem.getOptimalPath(agv.logicalNode, task.pickup);
                toDropoff = obj.agvSystem.getOptimalPath(task.pickup, task.dropoff);
                candidatePath = [toPickup, toDropoff(2:end)];

                % Calcola overlap embedding per ogni percorso attivo
                if isempty(activePaths)
                    congestionEmbs(:, i) = zeros(obj.embeddingDim, 1);
                    continue;
                end

                overlapEmbeddings = [];
                numValidOverlaps = 0;

                for j = 1:length(activePaths)
                    activePath = activePaths{j};
                    activeAGVRef = obj.agvSystem.AGVs(activeAGVIndices(j));

                    % Trova nodi overlap
                    overlapNodes = intersect(candidatePath, activePath);

                    if ~isempty(overlapNodes)
                        % Calcola overlap con riferimenti AGV per gestire movimento
                        overlapEmb = obj.computeOverlapEmbeddingForPath(...
                            candidatePath, activePath, overlapNodes, [], activeAGVRef);
                        overlapEmbeddings = [overlapEmbeddings, overlapEmb];
                        numValidOverlaps = numValidOverlaps + 1;
                    end
                end

                % Media degli overlap embeddings
                if numValidOverlaps > 0
                    congestionEmbs(:, i) = mean(overlapEmbeddings, 2);
                else
                    congestionEmbs(:, i) = zeros(obj.embeddingDim, 1);
                end
            end
        end
        
        function metrics = computeSystemMetrics(obj)
            % Calcola metriche sistema (throughput, conflitti, tempo)

            % CALCOLA METRICHE SISTEMA CON NORMALIZZAZIONE THROUGHPUT
            % Restituisce [throughputNorm, conflictRate, timeElapsed]
            %
            % OUTPUT:
            %   metrics(1) = throughputNorm ∈ [0, maxThroughputClip]
            %       - 0.0 = sistema fermo
            %       - 1.0 = target raggiunto
            %       - >1.0 = superamento target
            %   metrics(2) = conflictRate [conflitti/s]
            %   metrics(3) = timeElapsed ∈ [0, 1]

            % === THROUGHPUT NORMALIZZATO ===
            throughputRaw = obj.agvSystem.currentThroughputValue;

            % Normalizzazione rispetto a target
            if throughputRaw >= 0 && obj.targetThroughput > 0
                throughputNorm = throughputRaw / obj.targetThroughput;
                throughputNorm = min(throughputNorm, obj.maxThroughputClip);
            else
                throughputNorm = 0.0;
            end
            
            % Tasso conflitto al momento della decisione
            conflictRate = 0;
            if ~isempty(obj.agvSystem.activeManeuvers)
                numActiveManeuvers = length(obj.agvSystem.activeManeuvers);
                numAGVsInConflict = numActiveManeuvers * 2;  % Ogni manovra coinvolge 2 AGV
                conflictRate = numAGVsInConflict / obj.agvSystem.nAGV;

                % Clamp a 1.0 (nel caso teorico impossibile di più manovre sovrapposte)
                conflictRate = min(conflictRate, 1.0);
            end
            
            % Tempo trascorso (normalizzato)
            timeElapsed = obj.agvSystem.currentTime / obj.agvSystem.maxTime;
            
            metrics = [throughputNorm, conflictRate, timeElapsed];
        end
        
        function taskSegment = computeTaskSegment(obj, agv)
            % Identifica fase corrente del task AGV
            % NORMALIZZATO nell'intervallo [0, 1]

            if isempty(agv.task)
                taskSegmentRaw = -1; % Idle
            elseif agv.logicalNode == agv.task.pickup && isempty(agv.residualRoute)
                taskSegmentRaw = 1; % A pickup (arrivato)
            elseif agv.logicalNode == agv.task.dropoff
                taskSegmentRaw = 2; % A dropoff (arrivato)
            elseif ~isempty(agv.residualRoute)
                % Determina se sta andando verso pickup o dropoff
                target = agv.residualRoute(end);
                if target == agv.task.pickup
                    taskSegmentRaw = 1; % Verso pickup
                elseif target == agv.task.dropoff
                    taskSegmentRaw = 2; % Verso dropoff
                else
                    taskSegmentRaw = 0; % Verso parcheggio
                end
            else
                taskSegmentRaw = 0; % Verso parcheggio o situazione ambigua
            end

            % ✅ NORMALIZZAZIONE: [-1, 2] → [0, 1]
            % -1 → 0.0 (idle)
            %  0 → 0.33 (verso parcheggio)
            %  1 → 0.66 (verso/a pickup)
            %  2 → 1.0 (verso/a dropoff)
            taskSegment = (taskSegmentRaw + 1) / 3;
        end
        
        function embedding = getNodeEmbedding(obj, nodeIdx)
            % Ottiene embedding di un nodo specifico
            
            if nodeIdx >= 1 && nodeIdx <= size(obj.nodeEmbeddings, 1)
                embedding = obj.nodeEmbeddings(nodeIdx,:);
            else
                warning('Indice nodo non valido: %d', nodeIdx);
                embedding = zeros(obj.embeddingDim, 1);
            end
        end

        % === SHUFFLING E MAPPING ===
        
        function [shuffledQueue, shuffleMapping] = applyShuffling(obj, originalQueue)
            % Applica shuffling per prevenire bias posizionale
            
            if ~obj.enableShuffling || strcmp(obj.operatingMode, 'RL_TESTING')
                % Nessun shuffling in testing o se disabilitato
                shuffledQueue = originalQueue;
                shuffleMapping = 1:length(originalQueue);
            else
                % Shuffling durante training
                numTasks = length(originalQueue);
                shuffleMapping = randperm(numTasks);
                shuffledQueue = originalQueue(shuffleMapping);
                
                if obj.debugMode
                    obj.logMessage(sprintf('🔀 Shuffling applicato: [%s]', ...
                        strjoin(arrayfun(@num2str, shuffleMapping, 'UniformOutput', false), ',')));
                end
            end
        end
        
        function originalPosition = mapPositionToOriginal(obj, shuffledPosition, shuffleMapping)
            % Mappa posizione shuffled a posizione originale
            
            if shuffledPosition >= 1 && shuffledPosition <= length(shuffleMapping)
                originalPosition = shuffleMapping(shuffledPosition);
            else
                error('Posizione shuffled non valida: %d', shuffledPosition);
            end
        end
        
        function valid = validateAction(obj, action, queueLength)
            % Valida azione dell'agente
            
            valid = (action >= 1) && (action <= queueLength) && (action <= obj.maxTasks);
        end
        
        % === GESTIONE ESPERIENZE ===
        
        function startExperience(obj, agvId, localFeatures, globalFeatures, top5Queue, shuffleMapping)
            % Avvia nuova esperienza per AGV
            
            experience = struct();
            experience.agvId = agvId;

            experience.timestampDecision = obj.agvSystem.currentTime;
            experience.agvPositionAtDecision = obj.agvSystem.AGVs(agvId).logicalNode;
            
            % Osservazioni al momento decisione
            experience.localFeatures = localFeatures;
            experience.globalFeatures = globalFeatures;
            experience.top5QueueSnapshot = top5Queue;
            experience.shuffleMapping = shuffleMapping;
            experience.actionMask = obj.createActionMask(top5Queue);
            
            % Campi da completare dopo decisione/completamento
            experience.action = [];
            experience.taskSelected = [];
            experience.originalPosition = [];
            experience.nextLocalFeatures = [];
            experience.nextGlobalFeatures = [];
            experience.nextTop5Queue = [];
            experience.reward = [];
            experience.globalReward = [];
            experience.localReward = [];
            experience.completedTask = [];
            experience.done = false;
            experience.taskCompleted = false;
            
            % Salva esperienza aperta
            obj.openExperiences(agvId) = experience;
            
            obj.logMessage(sprintf('📝 Esperienza avviata per AGV %d', agvId));
        end

        function recordDecision(obj, agvId, selectedPosition, selectedTask, originalPosition)
            % Registra decisione presa dall'agente

            if obj.openExperiences.isKey(agvId)
                experience = obj.openExperiences(agvId);
                experience.action = selectedPosition;
                experience.taskSelected = selectedTask;
                experience.originalPosition = originalPosition;

                obj.openExperiences(agvId) = experience;

                obj.logMessage(sprintf('🎯 Decisione registrata AGV %d: pos %d → task %d', ...
                    agvId, selectedPosition, selectedTask.sequentialId));
            end
        end

        function completeExperience(obj, agvId, nextLocalFeatures, nextGlobalFeatures, ...
                completedTask, isDone)
            % COMPLETA ESPERIENZA CON COMPONENTI RAW E CALCOLA REWARD

            if obj.openExperiences.isKey(agvId)
                experience = obj.openExperiences(agvId);

                % Osservazioni next
                experience.nextLocalFeatures = nextLocalFeatures;
                experience.nextGlobalFeatures = nextGlobalFeatures;
                experience.nextTop5Queue = obj.agvSystem.getTop5TaskQueue();

                % ========== CALCOLA COMPONENTI REWARD ==========
                rewardComponents = obj.computeRewardComponents(agvId, experience, completedTask);
                experience.rawComponents = rewardComponents;

                % ========== CALCOLA REWARD NORMALIZZATE ==========
                [totalReward, globalReward, localReward] = obj.computeNormalizedReward(rewardComponents);

                experience.reward = totalReward;
                experience.globalReward = globalReward;
                experience.localReward = localReward;

                % Completamento
                experience.timestampCompletion = obj.agvSystem.currentTime;
                experience.completedTask = completedTask;
                experience.taskCompleted = true;
                experience.done = isDone;

                % Sposta in buffer
                obj.completedExperiencesByAGV{agvId} = ...
                    [obj.completedExperiencesByAGV{agvId}, experience];
                obj.openExperiences.remove(agvId);
                obj.totalExperienceCount = obj.totalExperienceCount + 1;

                obj.manageBufferSizeForAGV(agvId);

                if obj.verboseLogging
                    obj.logMessage(sprintf('✅ AGV %d: Reward=%.4f', agvId, totalReward));
                end
            end
        end

        function cancelExperience(obj, agvId)
            % Cancella esperienza aperta (in caso di fallback IDRR)
            
            if obj.openExperiences.isKey(agvId)
                obj.openExperiences.remove(agvId);
                obj.logMessage(sprintf('❌ Esperienza cancellata per AGV %d', agvId));
            end
        end
        
        function manageBufferSizeForAGV(obj, agvId)
            % Gestisce dimensione buffer per AGV specifico

            if length(obj.completedExperiencesByAGV{agvId}) > obj.maxBufferSizePerAGV
                % Rimuovi esperienze più vecchie (FIFO)
                numToRemove = length(obj.completedExperiencesByAGV{agvId}) - obj.maxBufferSizePerAGV;
                obj.completedExperiencesByAGV{agvId}(1:numToRemove) = [];

                if obj.debugMode
                    obj.logMessage(sprintf('🗑️ AGV %d: rimossi %d esperienze vecchie', agvId, numToRemove));
                end
            end
        end

        function totalCount = getTotalExperienceCount(obj)
            % Conta esperienze totali da tutti gli AGV
            totalCount = 0;
            for i = 1:obj.maxAGVs
                totalCount = totalCount + length(obj.completedExperiencesByAGV{i});
            end
        end

        function experiences = getAllCompletedExperiences(obj)
            % Raccoglie tutte le esperienze da tutti gli AGV
            experiences = [];
            for i = 1:obj.maxAGVs
                if ~isempty(obj.completedExperiencesByAGV{i})
                    experiences = [experiences, obj.completedExperiencesByAGV{i}];
                end
            end
        end

        function experiences = getExperiencesByAGV(obj, agvId, maxCount)
            % Restituisce esperienze per AGV specifico
            if nargin < 3
                maxCount = length(obj.completedExperiencesByAGV{agvId});
            end

            numToReturn = min(maxCount, length(obj.completedExperiencesByAGV{agvId}));

            if numToReturn > 0
                experiences = obj.completedExperiencesByAGV{agvId}(1:numToReturn);
            else
                experiences = [];
            end
        end
          
        % === INTERFACCIA PER CLASSE 3 ===

        function experiences = getCompletedExperiences(obj, maxCount, samplingMode)
            % Restituisce esperienze complete per training con sampling intelligente

            if nargin < 3, samplingMode = 'balanced'; end % 'balanced', 'random', 'recent'

            allExperiences = obj.getAllCompletedExperiences();

            if nargin < 2
                maxCount = length(allExperiences);
            end

            numToReturn = min(maxCount, length(allExperiences));

            if numToReturn > 0
                switch samplingMode
                    case 'balanced'
                        % Campionamento bilanciato da tutti gli AGV
                        experiences = obj.getBalancedSample(numToReturn);
                    case 'recent'
                        % Esperienze più recenti
                        [~, sortIdx] = sort([allExperiences.timestampCompletion], 'descend');
                        experiences = allExperiences(sortIdx(1:numToReturn));
                    otherwise
                        % Random sampling
                        randIdx = randperm(length(allExperiences), numToReturn);
                        experiences = allExperiences(randIdx);
                end

                obj.logMessage(sprintf('📦 Fornite %d esperienze (%s sampling)', numToReturn, samplingMode));
            else
                experiences = [];
            end
        end

        function experiences = getBalancedSample(obj, totalCount)
            % Campionamento bilanciato da tutti gli AGV
            experiences = [];
            agvCounts = zeros(obj.maxAGVs, 1);

            % Conta esperienze per AGV
            for i = 1:obj.maxAGVs
                agvCounts(i) = length(obj.completedExperiencesByAGV{i});
            end

            % Distribuisci uniformemente
            activeAGVs = find(agvCounts > 0);
            if isempty(activeAGVs)
                return;
            end

            expPerAGV = floor(totalCount / length(activeAGVs));
            remaining = totalCount - expPerAGV * length(activeAGVs);

            for i = activeAGVs'
                % Esperienze base per questo AGV
                numFromThisAGV = min(expPerAGV, agvCounts(i));

                % Aggiungi remainder se necessario
                if remaining > 0 && agvCounts(i) > expPerAGV
                    numFromThisAGV = numFromThisAGV + 1;
                    remaining = remaining - 1;
                end

                if numFromThisAGV > 0
                    agvExps = obj.getExperiencesByAGV(i, numFromThisAGV);
                    experiences = [experiences, agvExps];
                end
            end
        end
        
        
        % === INTERFACCIA CONFIGURAZIONE ===
        
        function setMAPPOAgent(obj, mappoAgent)
            % Imposta agente MAPPO
            
            obj.mappoAgent = mappoAgent;
            obj.logMessage('🤖 Agente MAPPO collegato');
        end
        
        function setOperatingMode(obj, mode)
            % Cambia modalità operativa
            
            validModes = {'IDRR', 'RL_TRAINING', 'RL_TESTING'};
            if ismember(mode, validModes)
                oldMode = obj.operatingMode;
                obj.operatingMode = mode;
                
                % Adatta shuffling alla modalità
                if strcmp(mode, 'RL_TESTING')
                    obj.enableShuffling = false;
                elseif strcmp(mode, 'RL_TRAINING')
                    obj.enableShuffling = true;
                end
                
                obj.logMessage(sprintf('🔄 Modalità cambiata: %s → %s', oldMode, mode));
            else
                error('Modalità non valida: %s', mode);
            end
        end
        
        function setRewardParameters(obj, lambda, alpha, beta, gamma, delta)
            % Imposta parametri reward
            
            if nargin >= 2, obj.lambda = lambda; end
            if nargin >= 3, obj.alpha = alpha; end
            if nargin >= 4, obj.beta = beta; end
            if nargin >= 5, obj.gamma = gamma; end
            if nargin >= 6, obj.delta = delta; end
            
            obj.logMessage(sprintf('⚖️ Parametri reward aggiornati: λ=%.2f, α=%.2f, β=%.2f, γ=%.2f, δ=%.2f', ...
                obj.lambda, obj.alpha, obj.beta, obj.gamma, obj.delta));
        end
        
        % === RESET E PULIZIA ===
                
        function resetTraining(obj)
            % Reset specifico per training

            % Pulisci tutti i buffer AGV
            for i = 1:obj.maxAGVs
                obj.completedExperiencesByAGV{i} = [];
            end

            obj.openExperiences = containers.Map('KeyType', 'int32', 'ValueType', 'any');

            % Reset contatori
            obj.totalDecisionRequests = 0;
            obj.totalExperiences = 0;
            obj.totalExperienceCount = 0;
            obj.positionMappingErrors = 0;

            obj.logMessage('🎯 Training reset: buffers e contatori azzerati');
        end

        % === MONITORING E DIAGNOSTICA ===

        function stats = getStatistics(obj)
            % Restituisce statistiche operative

            stats = struct();
            stats.totalDecisionRequests = obj.totalDecisionRequests;
            stats.totalExperiences = obj.totalExperiences;

            % CORREZIONE: Usa getTotalExperienceCount() per contare da completedExperiencesByAGV
            stats.completedExperiences = obj.getTotalExperienceCount();

            stats.openExperiences = obj.openExperiences.Count;
            stats.positionMappingErrors = obj.positionMappingErrors;
            stats.operatingMode = obj.operatingMode;
            stats.shufflingEnabled = obj.enableShuffling;

            % Calcola tasso successo
            if obj.totalDecisionRequests > 0
                stats.successRate = obj.totalExperiences / obj.totalDecisionRequests;
            else
                stats.successRate = 0;
            end
        end

        function printStatistics(obj)
            % Stampa statistiche dettagliate

            stats = obj.getStatistics();

            % fprintf('\n=== STATISTICHE AGV ENVIRONMENT ===\n');
            % fprintf('Modalità operativa: %s\n', stats.operatingMode);
            % fprintf('Shuffling abilitato: %s\n', string(stats.shufflingEnabled));
            % fprintf('Decisioni richieste: %d\n', stats.totalDecisionRequests);
            % fprintf('Esperienze completate: %d\n', stats.totalExperiences);
            % fprintf('Esperienze in buffer: %d\n', stats.completedExperiences);
            % fprintf('Esperienze aperte: %d\n', stats.openExperiences);
            % fprintf('Errori mapping: %d\n', stats.positionMappingErrors);
            % fprintf('Tasso successo: %.1f%%\n', stats.successRate * 100);

            % CORREZIONE: Usa getAllCompletedExperiences() per ottenere tutte le esperienze
            allCompletedExperiences = obj.getAllCompletedExperiences();

            % Statistiche reward se disponibili
            if ~isempty(allCompletedExperiences)
                rewards = [allCompletedExperiences.reward];
                %fprintf('Reward medio: %.3f ± %.3f\n', mean(rewards), std(rewards));
                %fprintf('Reward range: [%.3f, %.3f]\n', min(rewards), max(rewards));

                % Statistiche aggiuntive per buffer separati
                %fprintf('\n--- BREAKDOWN PER AGV ---\n');
                for agvId = 1:obj.maxAGVs
                    agvExpCount = length(obj.completedExperiencesByAGV{agvId});
                    if agvExpCount > 0
                        agvExperiences = obj.completedExperiencesByAGV{agvId};
                        agvRewards = [agvExperiences.reward];
                        %fprintf('AGV %d: %d esperienze, reward medio: %.3f\n', ...
                        %    agvId, agvExpCount, mean(agvRewards));
                    end
                end
            end

            %fprintf('================================\n');
        end

        % Statistiche dettagliate per AGV specifico
        function printAGVStatistics(obj, agvId)
            % Stampa statistiche per AGV specifico

            if agvId < 1 || agvId > obj.maxAGVs
                error('AGV ID non valido: %d (range: 1-%d)', agvId, obj.maxAGVs);
            end

            agvExperiences = obj.completedExperiencesByAGV{agvId};
            expCount = length(agvExperiences);

            %fprintf('\n=== STATISTICHE AGV %d ===\n', agvId);
            %fprintf('Esperienze nel buffer: %d/%d\n', expCount, obj.maxBufferSizePerAGV);

            if expCount > 0
                rewards = [agvExperiences.reward];
                globalRewards = [agvExperiences.globalReward];
                localRewards = [agvExperiences.localReward];

                %fprintf('Reward totali: %.3f ± %.3f [%.3f, %.3f]\n', ...
                %    mean(rewards), std(rewards), min(rewards), max(rewards));
                %fprintf('Reward globali: %.3f ± %.3f\n', mean(globalRewards), std(globalRewards));
                %fprintf('Reward locali: %.3f ± %.3f\n', mean(localRewards), std(localRewards));

                % Task completati per tipo se disponibile
                if isfield(agvExperiences, 'taskSelected')
                    taskTypes = [];
                    for i = 1:expCount
                        if ~isempty(agvExperiences(i).taskSelected) && isfield(agvExperiences(i).taskSelected, 'taskTypeId')
                            taskTypes(end+1) = agvExperiences(i).taskSelected.taskTypeId;
                        end
                    end

                    if ~isempty(taskTypes)
                        uniqueTypes = unique(taskTypes);
                        %fprintf('Task types completati: ');
                        for typeId = uniqueTypes
                            count = sum(taskTypes == typeId);
                            %fprintf('%d(×%d) ', typeId, count);
                        end
                        %fprintf('\n');
                    end
                end

                % Timestamp range
                if isfield(agvExperiences, 'timestampDecision')
                    timestamps = [agvExperiences.timestampDecision];
                    %fprintf('Periodo attività: %.1fs - %.1fs (durata: %.1fs)\n', ...
                        %min(timestamps), max(timestamps), max(timestamps) - min(timestamps));
                end
            else
                %fprintf('Nessuna esperienza registrata per questo AGV\n');
            end

            %fprintf('========================\n');
        end

        % Summary completo del buffer system
        function printBufferSummary(obj)
            % Stampa summary del sistema di buffer esperienze

            %fprintf('\n=== BUFFER SYSTEM SUMMARY ===\n');

            totalCapacity = obj.maxBufferSizePerAGV * obj.maxAGVs;
            totalUsed = obj.getTotalExperienceCount();

            % fprintf('Capacità totale: %d esperienze\n', totalCapacity);
            % fprintf('Esperienze memorizzate: %d (%.1f%%)\n', totalUsed, (totalUsed/totalCapacity)*100);
            % fprintf('Buffer size per AGV: %d\n', obj.maxBufferSizePerAGV);
            % fprintf('Esperienze aperte: %d\n', obj.openExperiences.Count);

            %fprintf('\n--- UTILIZZO PER AGV ---\n');
            for agvId = 1:obj.maxAGVs
                agvCount = length(obj.completedExperiencesByAGV{agvId});
                utilizationPct = (agvCount / obj.maxBufferSizePerAGV) * 100;

                if agvCount > 0
                    %fprintf('AGV %d: %d/%d (%.1f%%) ', agvId, agvCount, obj.maxBufferSizePerAGV, utilizationPct);

                    % Indica se buffer è pieno
                    if agvCount == obj.maxBufferSizePerAGV
                        %fprintf('[FULL]');
                    elseif utilizationPct > 80
                        %fprintf('[HIGH]');
                    end
                    fprintf('\n');
                end
            end

            % Statistiche temporali se disponibili
            allExperiences = obj.getAllCompletedExperiences();
            if ~isempty(allExperiences) && isfield(allExperiences, 'timestampDecision')
                timestamps = [allExperiences.timestampDecision];
                %fprintf('\nRange temporale: %.1fs - %.1fs\n', min(timestamps), max(timestamps));

                % Rate di generazione esperienze
                timeSpan = max(timestamps) - min(timestamps);
                if timeSpan > 0
                    experienceRate = length(allExperiences) / timeSpan;
                    %fprintf('Rate generazione: %.2f exp/s\n', experienceRate);
                end
            end

            %fprintf('=============================\n');
        end

        % Reset con statistiche
        function resetBuffersWithStats(obj)
            % Reset buffer con stampa statistiche pre-reset

            %fprintf('\n=== RESET BUFFERS ===\n');

            % Statistiche pre-reset
            totalExpBefore = obj.getTotalExperienceCount();
            %fprintf('Esperienze totali prima del reset: %d\n', totalExpBefore);

            if totalExpBefore > 0
                allExp = obj.getAllCompletedExperiences();
                rewards = [allExp.reward];
                %fprintf('Reward medio sessione: %.3f ± %.3f\n', mean(rewards), std(rewards));
            end

            % Esegui reset
            obj.resetTraining();

            %fprintf('✅ Buffer resettati (esperienze aperte: %d)\n', obj.openExperiences.Count);
            %fprintf('====================\n');
        end
        
        function validateIntegrity(obj)
            % Valida integrità interna del sistema
            
            issues = {};
            
            % Verifica embeddings
            if isempty(obj.nodeEmbeddings)
                issues{end+1} = 'Node embeddings non caricati';
            end
            
            % Verifica dimensioni features
            if obj.globalFeatureDim ~= 241
                issues{end+1} = sprintf('Dimensione features globali incorretta: %d != 241', obj.globalFeatureDim);
            end
            
            if obj.localFeatureDim ~= 98
                issues{end+1} = sprintf('Dimensione features locali incorretta: %d != 98', obj.localFeatureDim);
            end
            
            % Verifica esperienze aperte orfane
            if obj.openExperiences.Count > 0
                openAGVs = cell2mat(obj.openExperiences.keys);
                for agvId = openAGVs
                    if agvId > obj.agvSystem.nAGV
                        issues{end+1} = sprintf('Esperienza aperta per AGV non esistente: %d', agvId);
                    end
                end
            end
            
            % Report problemi
            if isempty(issues)
                obj.logMessage('✅ Validazione integrità: nessun problema trovato');
            else
                obj.logMessage('❌ Problemi integrità trovati:');
                for i = 1:length(issues)
                    obj.logMessage(sprintf('  - %s', issues{i}));
                end
            end
        end
        
        % === UTILITY LOGGING ===
        
        function logMessage(obj, message)
            % Log messaggio se verbose logging abilitato
            
            if obj.verboseLogging
                timestamp = datestr(now, 'HH:MM:SS.FFF');
                %fprintf('[%s ENV] %s\n', timestamp, message);
            end
        end
        
        function setVerboseLogging(obj, enable)
            % Abilita/disabilita logging verboso
            
            obj.verboseLogging = enable;
            obj.logMessage(sprintf('Verbose logging %s', obj.ternary(enable, 'abilitato', 'disabilitato')));
        end
        
        function setDebugMode(obj, enable)
            % Abilita/disabilita debug mode
            
            obj.debugMode = enable;
            obj.logMessage(sprintf('Debug mode %s', obj.ternary(enable, 'abilitato', 'disabilitato')));
        end
    end
    
    methods (Access = private)
        % === UTILITY FUNCTIONS ===

        function numTransferred = transferExperiencesToAgent(obj)
            % TRASFERISCE ESPERIENZE DA ENVIRONMENT A MAPPO_AGENT

            numTransferred = 0;

            % Per ogni AGV
            for agvId = 1:obj.maxAGVs
                agvExperiences = obj.completedExperiencesByAGV{agvId};

                if isempty(agvExperiences)
                    continue;
                end

                % Trasferisci ogni esperienza
                for i = 1:length(agvExperiences)
                    exp = agvExperiences(i);

                    % ✅ Chiama direttamente storeExperience di MAPPO_Agent
                    % con TUTTI i parametri necessari incluso actionMask
                    obj.mappoAgent.storeExperience(...
                        agvId, ...
                        exp.localFeatures, ...
                        exp.globalFeatures, ...
                        exp.action, ...
                        exp.reward, ...
                        exp.nextLocalFeatures, ...
                        exp.nextGlobalFeatures, ...
                        exp.done, ...
                        exp.actionMask ...  % ✅ CRITICO: passa mask
                        );

                    numTransferred = numTransferred + 1;
                end
            end
        end
        
        function result = ternary(~, condition, trueValue, falseValue)
            % Operatore ternario helper
            if condition
                result = trueValue;
            else  
                result = falseValue;
            end
        end
    end
end