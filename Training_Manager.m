classdef Training_Manager < handle
    % TRAINING_MANAGER - Orchestrazione completa ciclo vita training MAPPO
    % Gestisce: raccolta dati, training, testing, checkpointing, early stopping
    %
    % DESIGN SCELTE PER IL TEST INTRA-TRAINING:
    %   - I task di test vengono generati con generateRandomTask() della classe
    %     (stesso meccanismo del training, puro random, nessuna diversity window).
    %     Questo evita mismatch di struct (id vs sequentialId) e mantiene la
    %     distribuzione dei task identica tra training e test.
    %   - numTestTasks=200 task in coda a t=0: la simulazione si ferma per
    %     timeout a maxTime=3600s, non al completamento dei task.
    %   - Seed fisso per episodio test (stesso per ogni checkpoint): permette di
    %     confrontare l'andamento tra sessioni di test diverse.
    %   - Metriche raccolte: productivity, avgExecutionTime, avgWaitingTime,
    %     avgDistance, totalConflicts (+ breakdown headon/intersection/pursuit/loop),
    %     totalManeuvers (+ res1/res2).

    properties (Access = public)
        %% COMPONENTI SISTEMA
        agvSystem
        agvEnvironment
        mappoAgent

        %% CONFIGURAZIONE TRAINING
        nAGV
        maxTasks            % Task per episodio training
        maxTime             % Timeout episodio [s]
        taskGenRate
        preGeneratedTasks   % Task generati a t=0 con generateRandomTask()

        % Parametri training
        episodesPerTraining      = 100
        miniBatchSize            = 64
        ppoEpochs                = 15
        minExperiencesForTraining = 2000

        % Parametri testing e checkpoint
        testEpisodesInterval = 500
        checkpointInterval   = 1000
        numTestEpisodes      = 10    % Episodi per test session
        numTestTasks         = 200   % Task per episodio test (per riempire 3600s)

        %% EARLY STOPPING (patience + significatività statistica, senza soglie arbitrarie)
        earlyStoppingEnabled          = true
        significanceLevel             = 0.05
        maxEpisodesWithoutImprovement = 1000

        %% METRICHE TRAINING
        trainingMetrics
        testingMetrics
        episodeData

        %% STATO TRAINING
        currentEpisode              = 0
        totalTrainingSteps          = 0
        episodeSinceLastTest        = 0
        episodeSinceLastImprovement = 0
        bestTestPerformance         = -inf   % Massimo grezzo osservato: pilota il salvataggio di best_model.mat
        bestTestSample              = []     % Campione (numTestEpisodes x 1) del checkpoint bestTestPerformance
        bestConfirmedPerformance    = -inf   % Ultimo miglioramento statisticamente confermato: pilota SOLO il reset della patience
        bestConfirmedSample         = []     % Campione (numTestEpisodes x 1) del checkpoint bestConfirmedPerformance
        convergenceAchieved         = false

        %% PATHS E SALVATAGGIO
        savePath = './training_results/'
        experimentName

        %% LOGGING
        verboseLogging = true
        logFile
    end

    methods
        % ================================================================
        % COSTRUTTORE
        % ================================================================
        function obj = Training_Manager(nAGV, maxTasks, maxTime, taskGenRate, varargin)
            p = inputParser;
            addRequired(p, 'nAGV',       @isnumeric);
            addRequired(p, 'maxTasks',    @isnumeric);
            addRequired(p, 'maxTime',     @isnumeric);
            addRequired(p, 'taskGenRate', @isnumeric);
            addParameter(p, 'preGeneratedTasks',   5,                            @isnumeric);
            addParameter(p, 'episodesPerTraining', 3,                            @isnumeric);
            addParameter(p, 'experimentName',      datestr(now,'yyyymmdd_HHMMSS'), @ischar);
            addParameter(p, 'verboseLogging',      true,                         @islogical);
            parse(p, nAGV, maxTasks, maxTime, taskGenRate, varargin{:});

            obj.nAGV                = p.Results.nAGV;
            obj.maxTasks            = p.Results.maxTasks;
            obj.maxTime             = p.Results.maxTime;
            obj.taskGenRate         = p.Results.taskGenRate;
            obj.preGeneratedTasks   = p.Results.preGeneratedTasks;
            obj.episodesPerTraining = p.Results.episodesPerTraining;
            obj.experimentName      = p.Results.experimentName;
            obj.verboseLogging      = p.Results.verboseLogging;

            obj.savePath = fullfile(obj.savePath, obj.experimentName);
            if ~exist(obj.savePath, 'dir'), mkdir(obj.savePath); end

            obj.initializeLogging();
            obj.initializeSystemComponents();
            obj.initializeMetricsStructures();

            obj.log('Training Manager inizializzato');
            obj.log(sprintf('Esperimento: %s', obj.experimentName));
            obj.log(sprintf('preGeneratedTasks=%d | episodesPerTraining=%d | numTestEpisodes=%d | numTestTasks=%d', ...
                obj.preGeneratedTasks, obj.episodesPerTraining, obj.numTestEpisodes, obj.numTestTasks));
        end

        % ================================================================
        % INIZIALIZZAZIONE COMPONENTI
        % ================================================================
        function initializeSystemComponents(obj)
            obj.log('Inizializzazione componenti sistema...');

            obj.agvSystem = AGV_System_IDRR_RL(obj.nAGV, obj.maxTime, ...
                obj.maxTasks, obj.taskGenRate, ...
                'enableVisualization', false, ...
                'operatingMode',       'RL_TRAINING', ...
                'verboseLogging',      false);

            obj.agvEnvironment = AGV_Environment(obj.agvSystem, ...
                'operatingMode',  'RL_TRAINING', ...
                'enableShuffling', true, ...
                'verboseLogging',  false, ...
                'debugMode',       false);

            obj.agvSystem.connectToEnvironment(obj.agvEnvironment);

            obj.mappoAgent = MAPPO_Agent(obj.nAGV, ...
                'learningRate', 3e-4, 'epsilon', 0.2, 'gamma', 0.99, ...
                'lambda', 0.95, 'valueCoeff', 0.5, 'entropyCoeff', 0.01, ...
                'maxGradNorm', 0.5, 'verboseLogging', false, 'debugMode', false);

            obj.mappoAgent.miniBatchSize  = obj.miniBatchSize;
            obj.mappoAgent.ppoEpochs      = obj.ppoEpochs;
            obj.agvEnvironment.mappoAgent = obj.mappoAgent;
            obj.mappoAgent.resetAgent();

            obj.log('Componenti sistema inizializzati');
        end

        % ================================================================
        % STRUTTURE METRICHE
        % Le testingMetrics raccolgono la media su numTestEpisodes per ogni
        % sessione di test. Ogni campo scalare è un vettore [1 x numTestSessions].
        % ================================================================
        function initializeMetricsStructures(obj)
            obj.trainingMetrics = struct(...
                'episodeRewards',            [], ...
                'episodeRewardsMA',          [], ...
                'policyLoss',                [], ...
                'valueLoss',                 [], ...
                'trainingSteps',             [], ...
                'episodeProductivity',       [], ...
                'episodeDuration',           [], ...
                'totalExperiencesCollected', 0);

            % Metriche per RL deterministico e IDRR: medie su numTestEpisodes
            testMetricFields = struct(...
                'productivity',     [], ...  % [task/h]
                'avgExecutionTime', [], ...  % [s]
                'avgWaitingTime',   [], ...  % [s]
                'avgDistance',      [], ...  % [m]
                'totalConflicts',   [], ...
                'headon',           [], ...
                'intersection',     [], ...
                'pursuit',          [], ...
                'loop',             [], ...
                'totalManeuvers',   [], ...
                'res1',             [], ...
                'res2',             []);

            obj.testingMetrics = struct(...
                'testEpisodes',              [], ...
                'isNewBest',                 [], ...
                'isSignificantImprovement',  [], ...
                'rlDeterministic',           testMetricFields, ...
                'idrr',                      testMetricFields, ...
                'improvements',              struct('deterministic', []));

            obj.episodeData = {};
        end

        % ================================================================
        % LOGGING
        % ================================================================
        function initializeLogging(obj)
            logPath    = fullfile(obj.savePath, 'training.log');
            obj.logFile = fopen(logPath, 'w');
            fprintf(obj.logFile, '=== TRAINING LOG ===\n');
            fprintf(obj.logFile, 'Timestamp: %s\n', datestr(now));
            fprintf(obj.logFile, '====================\n\n');
        end

        function log(obj, message)
            ts  = datestr(now, 'HH:MM:SS');
            msg = sprintf('[%s] %s', ts, message);
            if obj.verboseLogging, fprintf('%s\n', msg); end
            if ~isempty(obj.logFile) && obj.logFile > 0
                fprintf(obj.logFile, '%s\n', msg);
            end
        end

        % ================================================================
        % MAIN TRAINING LOOP
        % ================================================================
        function runTraining(obj, numEpisodes)
            obj.log('═══════════════════════════════════════');
            obj.log(sprintf('INIZIO TRAINING: %d episodi', numEpisodes));
            obj.log('═══════════════════════════════════════');

            epSinceTraining = 0;

            for ep = 1:numEpisodes
                obj.currentEpisode = ep;
                fprintf('Episodio: %d\n', ep);

                % ===== FASE 1: EPISODIO TRAINING =====
                episodeResult = obj.runSingleEpisode(ep);

                % Salta se scartato per deadlock
                if isempty(episodeResult)
                    obj.log(sprintf('>>> Episodio %d saltato (DEADLOCK)', ep));
                    continue;
                end

                % Accumula metriche episodio valido
                obj.episodeData{end+1} = episodeResult;
                epSinceTraining        = epSinceTraining + 1;

                obj.trainingMetrics.episodeRewards(end+1)      = episodeResult.totalReward;
                obj.trainingMetrics.episodeProductivity(end+1) = episodeResult.productivity;
                obj.trainingMetrics.episodeDuration(end+1)     = episodeResult.duration;
                obj.trainingMetrics.totalExperiencesCollected  = ...
                    obj.trainingMetrics.totalExperiencesCollected + episodeResult.numExperiences;

                w = min(10, length(obj.trainingMetrics.episodeRewards));
                obj.trainingMetrics.episodeRewardsMA(end+1) = ...
                    mean(obj.trainingMetrics.episodeRewards(end-w+1:end));

                % ===== FASE 2: TRAINING STEP =====
                if epSinceTraining >= obj.episodesPerTraining
                    obj.log(sprintf('[Ep %d] TRAINING STEP', ep));
                    ok = obj.performTrainingStep();

                    if ok
                        epSinceTraining          = 0;
                        obj.totalTrainingSteps   = obj.totalTrainingSteps + 1;
                        obj.episodeSinceLastTest = obj.episodeSinceLastTest + obj.episodesPerTraining;

                        obj.trainingMetrics.policyLoss(end+1)    = obj.mappoAgent.PolicyLoss;
                        obj.trainingMetrics.valueLoss(end+1)     = obj.mappoAgent.ValueLoss;
                        obj.trainingMetrics.trainingSteps(end+1) = obj.totalTrainingSteps;
                    end
                end

                % ===== FASE 3: TEST SESSION =====
                if obj.episodeSinceLastTest >= obj.testEpisodesInterval
                    obj.log(sprintf('[Ep %d] TEST SESSION', ep));
                    obj.runTestSession();
                    obj.episodeSinceLastTest = 0;

                    % Sicurezza: verifica buffer vuoto dopo test
                    totalBuf = 0;
                    for agvId = 1:obj.nAGV
                        totalBuf = totalBuf + length(obj.mappoAgent.agvExperienceBuffers{agvId});
                    end
                    if totalBuf > 0
                        obj.log(sprintf('  WARNING: buffer non vuoto dopo test (%d exp), pulizia', totalBuf));
                        obj.mappoAgent.clearBuffers();
                    end

                    if obj.earlyStoppingEnabled && obj.checkEarlyStopping()
                        obj.log('EARLY STOPPING attivato');
                        break;
                    end
                end

                % ===== FASE 4: CHECKPOINT =====
                if mod(ep, obj.checkpointInterval) == 0
                    obj.log(sprintf('[Ep %d] CHECKPOINT', ep));
                    obj.saveCheckpoint(ep);
                end

                if mod(ep, 10) == 0
                    obj.printProgressReport(ep, numEpisodes);
                end
            end

            % ===== FINALIZZAZIONE =====
            obj.log('TRAINING COMPLETATO - test finale...');
            obj.runTestSession();
            obj.saveFinalResults();
            obj.plotTrainingMetrics();
            obj.log('Done.');
        end

        % ================================================================
        % SINGOLO EPISODIO TRAINING
        % - 10 task generati con generateRandomTask() a t=0
        % - Restanti (fino a maxTasks=21) via Poisson durante la simulazione
        % ================================================================
        function episodeResult = runSingleEpisode(obj, episodeIdx)
            obj.agvSystem.resetSystem();
            obj.agvSystem.resetForNewEpisode();
            obj.agvSystem.setOperatingMode('RL_TRAINING');
            obj.agvEnvironment.setOperatingMode('RL_TRAINING');

            % Genera preGeneratedTasks task a t=0 con generateRandomTask()
            % (stesso meccanismo del sistema reale, puro random)
            for i = 1:obj.preGeneratedTasks
                obj.agvSystem.generateRandomTask();
                obj.agvSystem.tasksGenerated = obj.agvSystem.tasksGenerated + 1;
            end

            obj.agvSystem.maxTime  = obj.maxTime;
            obj.agvSystem.maxTasks = obj.maxTasks;
            obj.agvSystem.enableDeadlockDetection = false;

            initialTime      = obj.agvSystem.currentTime;
            initialCompleted = length(obj.agvSystem.completedTasks);

            obj.agvSystem.initializeAGVs();
            
            try
                obj.agvSystem.runSimulation();
            catch ME
                if strcmp(ME.identifier, 'AGV_DEADLOCK:AllWaiting')
                    obj.log(sprintf('  Episodio %d DEADLOCK: %s', episodeIdx, ME.message));
                    obj.mappoAgent.clearBuffers();
                    obj.agvEnvironment.resetTraining();
                    episodeResult = [];
                    return;
                else
                    rethrow(ME);
                end
            end

            finalTime      = obj.agvSystem.currentTime;
            tasksCompleted = length(obj.agvSystem.completedTasks) - initialCompleted;
            duration       = finalTime - initialTime;

            numExperiences = obj.agvEnvironment.getTotalExperienceCount();

            totalReward = 0;
            for agvId = 1:obj.nAGV
                exps = obj.agvEnvironment.completedExperiencesByAGV{agvId};
                if ~isempty(exps)
                    totalReward = totalReward + sum([exps.reward]);
                end
            end

            episodeResult = struct(...
                'episodeIdx',    episodeIdx, ...
                'tasksCompleted', tasksCompleted, ...
                'duration',      duration, ...
                'productivity',  tasksCompleted / (duration / 3600), ...
                'numExperiences', numExperiences, ...
                'totalReward',   totalReward);

            obj.agvEnvironment.resetTraining();
        end

        % ================================================================
        % TRAINING STEP MAPPO
        % ================================================================
        function success = performTrainingStep(obj)
            totalBuf = 0;
            for agvId = 1:obj.nAGV
                totalBuf = totalBuf + length(obj.mappoAgent.agvExperienceBuffers{agvId});
            end
            obj.log(sprintf('  Buffer: %d esperienze', totalBuf));

            if totalBuf >= obj.minExperiencesForTraining
                success = obj.mappoAgent.trainStep(obj.minExperiencesForTraining);
                if success
                    obj.log(sprintf('  OK: Policy=%.4f, Value=%.4f', ...
                        obj.mappoAgent.PolicyLoss, obj.mappoAgent.ValueLoss));
                    obj.mappoAgent.clearBuffers();

                    totalAfter = 0;
                    for agvId = 1:obj.nAGV
                        totalAfter = totalAfter + length(obj.mappoAgent.agvExperienceBuffers{agvId});
                    end
                    if totalAfter > 0
                        obj.log(sprintf('  WARNING: buffer residuo (%d exp)', totalAfter));
                    end
                else
                    obj.log('  Training step fallito');
                end
            else
                obj.log(sprintf('  Buffer insufficiente (%d/%d)', totalBuf, obj.minExperiencesForTraining));
                success = false;
            end
        end

        % ================================================================
        % TEST SESSION
        % Esegue numTestEpisodes episodi da 3600s per RL det e IDRR.
        % I task sono generati con generateRandomTask() (puro random, stesso
        % meccanismo del training). Seed fisso per episodio: le stesse
        % configurazioni task si ripetono ad ogni checkpoint, permettendo
        % confronto diretto tra sessioni di test diverse.
        %
        % SELEZIONE DEL BEST CHECKPOINT:
        % Un checkpoint diventa il nuovo "best" solo se (i) la sua
        % produttività media RL Det. supera quella del best precedente E
        % (ii) la differenza è statisticamente significativa rispetto al
        % campione a numTestEpisodes episodi del best precedente, tramite
        % Welch's t-test a una coda. Nessuna soglia di miglioramento
        % percentuale arbitraria è utilizzata (cfr. Henderson et al.,
        % "Deep Reinforcement Learning that Matters", AAAI 2018, sull'uso
        % di test di significatività per confrontare policy RL).
        % ================================================================
        function runTestSession(obj)
            obj.log('────────────────────────────────────────');
            obj.log(sprintf('TEST SESSION: %d ep x %ds | %d task/ep', ...
                obj.numTestEpisodes, obj.maxTime, obj.numTestTasks));

            nM = obj.numTestEpisodes;
            rlFields   = {'productivity','avgExecutionTime','avgWaitingTime', ...
                          'avgDistance','totalConflicts','headon','intersection', ...
                          'pursuit','loop','totalManeuvers','res1','res2'};

            rlData   = zeros(nM, length(rlFields));
            idrrData = zeros(nM, length(rlFields));

            obj.log('  Test RL (deterministico)...');
            for i = 1:nM
                seed   = 9000 + i;
                result = obj.runSingleTestEpisode('RL_TESTING', seed, true);
                rlData(i, :) = obj.extractMetricRow(result, rlFields);
            end

            obj.log('  Test IDRR (baseline)...');
            for i = 1:nM
                seed   = 9000 + i;
                result = obj.runSingleTestEpisode('IDRR', seed, false);
                idrrData(i, :) = obj.extractMetricRow(result, rlFields);
            end

            obj.agvSystem.setOperatingMode('RL_TRAINING');
            obj.agvEnvironment.setOperatingMode('RL_TRAINING');

            rlMean   = mean(rlData,   1);
            idrrMean = mean(idrrData, 1);
            impDet   = (rlMean(1) - idrrMean(1)) / max(idrrMean(1), 1e-9) * 100;

            [~, pDet] = ttest2(rlData(:,1), idrrData(:,1));

            % --- (1) BEST GREZZO: sempre aggiornato se la media migliora ---
            isNewRawBest = isempty(obj.bestTestSample) || rlMean(1) > obj.bestTestPerformance;

            % --- (2) BEST CONFERMATO: test t APPAIATO (stesso seed per indice i),
            % usato SOLO per il reset della patience dell'early stopping ---
            isSignificantImprovement = false;
            if isempty(obj.bestConfirmedSample)
                isSignificantImprovement = true;
                pConfirmed = NaN;
            else
                differenze = rlData(:,1) - obj.bestConfirmedSample;
                [~, pConfirmed] = ttest(differenze, 0, 'Tail', 'right');
                if rlMean(1) > obj.bestConfirmedPerformance && pConfirmed < obj.significanceLevel
                    isSignificantImprovement = true;
                end
            end

            obj.testingMetrics.testEpisodes(end+1)               = obj.currentEpisode;
            obj.testingMetrics.isNewBest(end+1)                  = isNewRawBest;
            obj.testingMetrics.isSignificantImprovement(end+1)   = isSignificantImprovement;
            obj.testingMetrics.improvements.deterministic(end+1) = impDet;

            for fi = 1:length(rlFields)
                f = rlFields{fi};
                obj.testingMetrics.rlDeterministic.(f)(end+1) = rlMean(fi);
                obj.testingMetrics.idrr.(f)(end+1)            = idrrMean(fi);
            end

            obj.log('  Risultati:');
            obj.log(sprintf('    IDRR:   %.2f task/h | wait=%.1fs | conflicts=%d (HO=%d INT=%d PU=%d) | R1=%d R2=%d', ...
                idrrMean(1), idrrMean(3), idrrMean(5), idrrMean(6), idrrMean(7), idrrMean(8), idrrMean(11), idrrMean(12)));
            obj.log(sprintf('    RL Det: %.2f task/h | wait=%.1fs | conflicts=%d (HO=%d INT=%d PU=%d) | R1=%d R2=%d', ...
                rlMean(1), rlMean(3), rlMean(5), rlMean(6), rlMean(7), rlMean(8), rlMean(11), rlMean(12)));
            obj.log(sprintf('    Delta vs IDRR: %+.2f%% (p=%.4f)', impDet, pDet));

            if isNewRawBest
                obj.bestTestPerformance = rlMean(1);
                obj.bestTestSample      = rlData(:, 1);
                obj.log(sprintf('    NUOVO BEST (grezzo): %.2f task/h -> salvato in best_model.mat', ...
                    obj.bestTestPerformance));
                obj.saveCheckpoint(obj.currentEpisode, true);
            end

            if isSignificantImprovement
                obj.bestConfirmedPerformance    = rlMean(1);
                obj.bestConfirmedSample         = rlData(:, 1);
                obj.episodeSinceLastImprovement = 0;
                obj.log(sprintf('    Miglioramento CONFERMATO (p vs. confermato precedente = %.4f)', pConfirmed));
            else
                obj.episodeSinceLastImprovement = ...
                    obj.episodeSinceLastImprovement + obj.testEpisodesInterval;
                obj.log(sprintf('    Nessun miglioramento confermato (p vs. confermato = %.4f)', pConfirmed));
            end

            obj.log('────────────────────────────────────────');
        end

        % ================================================================
        % SINGOLO EPISODIO DI TEST
        % - Genera numTestTasks task con generateRandomTask() dopo rng(seed)
        % - task struct identico a quello del training: nessun mismatch
        % - maxTasks = numTestTasks: blocca Poisson, la sim gira fino a 3600s
        % ================================================================
        function result = runSingleTestEpisode(obj, mode, seed, useDeterministic)
            obj.agvSystem.resetSystem();
            obj.agvSystem.resetForNewEpisode();
            obj.agvSystem.setOperatingMode(mode);

            if strcmp(mode, 'RL_TESTING')
                obj.agvEnvironment.resetTraining();
                obj.agvEnvironment.setOperatingMode(mode);
                if useDeterministic
                    obj.mappoAgent.deterministicMode = true;
                end
            end

            % Genera numTestTasks task con generateRandomTask() in modo riproducibile.
            % Non usiamo la diversity window: puro random, coerente con il training.
            rng(seed);
            for t = 1:obj.numTestTasks
                obj.agvSystem.generateRandomTask();
            end
            % Imposta tasksGenerated = numTestTasks per bloccare generazione Poisson:
            % la simulazione si fermerà per timeout a maxTime=3600s.
            obj.agvSystem.tasksGenerated = obj.numTestTasks;
            obj.agvSystem.maxTime        = obj.maxTime;    % 3600s
            obj.agvSystem.maxTasks       = obj.numTestTasks;

            obj.agvSystem.initializeAGVs();
            obj.agvSystem.runSimulation();

            % Raccoglie tutte le metriche rilevanti
            fm = obj.agvSystem.metrics.finalMetrics;
            cc = obj.agvSystem.metrics.conflictCounts;
            mc = obj.agvSystem.metrics.maneuverCounts;

            result = struct(...
                'productivity',     fm.productivity, ...
                'avgExecutionTime', fm.avgExecutionTime, ...
                'avgWaitingTime',   fm.avgWaitingTime, ...
                'avgDistance',      fm.avgDistancePerAGV, ...
                'totalConflicts',   fm.totalConflicts, ...
                'headon',           cc.headon, ...
                'intersection',     cc.intersection, ...
                'pursuit',          cc.pursuit, ...
                'loop',             cc.loop, ...
                'totalManeuvers',   fm.totalManeuvers, ...
                'res1',             mc.res1, ...
                'res2',             mc.res2);

            if strcmp(mode, 'RL_TESTING') && useDeterministic
                obj.mappoAgent.deterministicMode = false;
            end

            obj.agvEnvironment.resetTraining();
        end

        % ================================================================
        % EARLY STOPPING
        % Criterio basato unicamente su patience: il training termina se
        % non viene registrato un NUOVO BEST statisticamente significativo
        % (vedi runTestSession) per maxEpisodesWithoutImprovement episodi
        % consecutivi. Meccanismo standard di early stopping basato su
        % patience (Prechelt, "Early Stopping - But When?", in Neural
        % Networks: Tricks of the Trade, Springer, 1998), senza soglie di
        % miglioramento percentuale arbitrarie.
        % ================================================================
        function shouldStop = checkEarlyStopping(obj)
            shouldStop = false;

            if obj.episodeSinceLastImprovement >= obj.maxEpisodesWithoutImprovement
                obj.log(sprintf('Early stopping: nessun miglioramento confermato da %d episodi', ...
                    obj.episodeSinceLastImprovement));
                obj.convergenceAchieved = true;
                shouldStop = true;
            end
        end

        % ================================================================
        % CHECKPOINT E SALVATAGGIO
        % ================================================================
        function saveCheckpoint(obj, episodeNum, isBest)
            if nargin < 3, isBest = false; end

            if isBest
                filename = fullfile(obj.savePath, 'best_model.mat');
            else
                filename = fullfile(obj.savePath, sprintf('checkpoint_ep%d.mat', episodeNum));
            end
            obj.mappoAgent.saveAgent(filename);

            mFile = fullfile(obj.savePath, sprintf('metrics_ep%d.mat', episodeNum));
            trainingMetrics = obj.trainingMetrics;
            testingMetrics  = obj.testingMetrics;
            save(mFile, 'trainingMetrics', 'testingMetrics');

            obj.log(sprintf('Checkpoint salvato: %s', filename));
        end

        function saveFinalResults(obj)
            finalFile       = fullfile(obj.savePath, 'final_results.mat');
            trainingMetrics = obj.trainingMetrics;
            testingMetrics  = obj.testingMetrics;
            episodeData     = obj.episodeData;
            config = struct(...
                'nAGV',                          obj.nAGV, ...
                'maxTasks',                       obj.maxTasks, ...
                'maxTime',                        obj.maxTime, ...
                'preGeneratedTasks',              obj.preGeneratedTasks, ...
                'numTestEpisodes',                obj.numTestEpisodes, ...
                'numTestTasks',                   obj.numTestTasks, ...
                'episodesPerTraining',            obj.episodesPerTraining, ...
                'miniBatchSize',                  obj.miniBatchSize, ...
                'ppoEpochs',                       obj.ppoEpochs, ...
                'significanceLevel',              obj.significanceLevel, ...
                'maxEpisodesWithoutImprovement',  obj.maxEpisodesWithoutImprovement);

            save(finalFile, 'trainingMetrics', 'testingMetrics', 'episodeData', 'config');
            obj.mappoAgent.saveAgent(fullfile(obj.savePath, 'final_model.mat'));
            obj.log(sprintf('Risultati finali salvati: %s', finalFile));
        end

        % ================================================================
        % PLOT METRICHE
        % Chiamato automaticamente a fine training. Può essere richiamato
        % manualmente in qualsiasi momento: trainingManager.plotTrainingMetrics()
        % ================================================================
        function plotTrainingMetrics(obj)
            set(groot,'defaultAxesTickLabelInterpreter','latex');
            set(groot,'defaultTextInterpreter','latex');
            set(groot,'defaultLegendInterpreter','latex');
            set(groot,'defaultAxesFontSize',12);

            fig = figure('Position',[100 100 1400 900],'Color','w');
            testEps = obj.testingMetrics.testEpisodes;
            eps_v   = 1:length(obj.trainingMetrics.episodeRewards);

            % --- 1: Reward per episodio + MA10 ---
            subplot(2,3,1);
            plot(eps_v, obj.trainingMetrics.episodeRewards, ...
                'Color',[0.75 0.75 0.75],'LineWidth',1,'DisplayName','Reward');
            hold on;
            plot(eps_v, obj.trainingMetrics.episodeRewardsMA, ...
                'b-','LineWidth',2,'DisplayName','MA-10');
            grid on;
            xlabel('Episode','FontSize',13); ylabel('Total Reward','FontSize',13);
            title('\textbf{Episode Reward}','FontSize',14); legend('Location','best');

            % --- 2: Policy Loss e Value Loss ---
            subplot(2,3,2);
            if ~isempty(obj.trainingMetrics.trainingSteps)
                yyaxis left;
                plot(obj.trainingMetrics.trainingSteps, obj.trainingMetrics.policyLoss, ...
                    'r-','LineWidth',2); ylabel('Policy Loss','FontSize',13);
                yyaxis right;
                plot(obj.trainingMetrics.trainingSteps, obj.trainingMetrics.valueLoss, ...
                    'b-','LineWidth',2); ylabel('Value Loss','FontSize',13);
                xlabel('Training Step','FontSize',13);
                title('\textbf{Training Losses}','FontSize',14); grid on;
            end

            % --- 3: Produttività test ---
            subplot(2,3,3);
            if ~isempty(testEps)
                plot(testEps, obj.testingMetrics.idrr.productivity, ...
                    'k--','LineWidth',2,'DisplayName','IDRR');
                hold on;
                plot(testEps, obj.testingMetrics.rlDeterministic.productivity, ...
                    'r-s','LineWidth',2,'MarkerSize',6,'DisplayName','RL Det');
                grid on;
                xlabel('Episode','FontSize',13); ylabel('Productivity [task/h]','FontSize',13);
                title('\textbf{Test Productivity}','FontSize',14); legend('Location','best');
            end

            % --- 4: Improvement % vs IDRR + marcatura nuovi best ---
            subplot(2,3,4);
            if ~isempty(testEps)
                plot(testEps, obj.testingMetrics.improvements.deterministic, ...
                    'r-o','LineWidth',2,'MarkerSize',6,'DisplayName','$\Delta\%$ vs IDRR');
                hold on;
                yline(0,'k--','LineWidth',1.5);
                bestIdx = logical(obj.testingMetrics.isNewBest);
                if any(bestIdx)
                    plot(testEps(bestIdx), obj.testingMetrics.improvements.deterministic(bestIdx), ...
                        'gp','MarkerSize',12,'MarkerFaceColor','g','DisplayName','Nuovo best');
                end
                grid on;
                xlabel('Episode','FontSize',13); ylabel('Improvement [\%]','FontSize',13);
                title('\textbf{RL Improvement over IDRR}','FontSize',14); legend('Location','best');
            end

            % --- 5: Conflitti IDRR vs RL ---
            subplot(2,3,5);
            if ~isempty(testEps)
                plot(testEps, obj.testingMetrics.idrr.totalConflicts, ...
                    'k--','LineWidth',2,'DisplayName','IDRR');
                hold on;
                plot(testEps, obj.testingMetrics.rlDeterministic.totalConflicts, ...
                    'r-s','LineWidth',2,'MarkerSize',6,'DisplayName','RL Det');
                grid on;
                xlabel('Episode','FontSize',13); ylabel('Total Conflicts','FontSize',13);
                title('\textbf{Test Conflicts}','FontSize',14); legend('Location','best');
            end

            % --- 6: Avg Waiting Time IDRR vs RL ---
            subplot(2,3,6);
            if ~isempty(testEps)
                plot(testEps, obj.testingMetrics.idrr.avgWaitingTime, ...
                    'k--','LineWidth',2,'DisplayName','IDRR');
                hold on;
                plot(testEps, obj.testingMetrics.rlDeterministic.avgWaitingTime, ...
                    'r-s','LineWidth',2,'MarkerSize',6,'DisplayName','RL Det');
                grid on;
                xlabel('Episode','FontSize',13); ylabel('Avg Waiting Time [s]','FontSize',13);
                title('\textbf{Test Avg Waiting Time}','FontSize',14); legend('Location','best');
            end

            sgtitle(sprintf('\\textbf{Training: %s}', strrep(obj.experimentName,'_','\_')), ...
                'FontSize',16);

            savefig(fig, fullfile(obj.savePath,'training_metrics.fig'));
            saveas(fig,  fullfile(obj.savePath,'training_metrics.png'));
            obj.log(sprintf('Plot salvato in %s', obj.savePath));
        end

        % ================================================================
        % PROGRESS REPORT
        % ================================================================
        function printProgressReport(obj, currentEp, totalEp)
            fprintf('\n═══════════════════════════════════════\n');
            fprintf('PROGRESS REPORT - Episode %d/%d\n', currentEp, totalEp);
            fprintf('═══════════════════════════════════════\n');

            if ~isempty(obj.trainingMetrics.episodeRewards)
                w = min(10, length(obj.trainingMetrics.episodeRewards));
                r = obj.trainingMetrics.episodeRewards(end-w+1:end);
                fprintf('Reward (last 10): %.2f +/- %.2f\n', mean(r), std(r));
            end
            if ~isempty(obj.trainingMetrics.policyLoss)
                fprintf('Policy Loss: %.4f | Value Loss: %.4f\n', ...
                    obj.trainingMetrics.policyLoss(end), obj.trainingMetrics.valueLoss(end));
            end
            fprintf('Training Steps: %d | Exp totali: %d\n', ...
                obj.totalTrainingSteps, obj.trainingMetrics.totalExperiencesCollected);

            if ~isempty(obj.testingMetrics.testEpisodes)
                fprintf('Test (ultimo):\n');
                fprintf('  IDRR:   %.2f task/h | conflicts=%d\n', ...
                    obj.testingMetrics.idrr.productivity(end), ...
                    obj.testingMetrics.idrr.totalConflicts(end));
                fprintf('  RL Det: %.2f task/h | conflicts=%d | delta=%+.2f%%\n', ...
                    obj.testingMetrics.rlDeterministic.productivity(end), ...
                    obj.testingMetrics.rlDeterministic.totalConflicts(end), ...
                    obj.testingMetrics.improvements.deterministic(end));
                fprintf('  Best:   %.2f task/h\n', obj.bestTestPerformance);
            end

            if obj.earlyStoppingEnabled
                fprintf('Early stopping: %d/%d ep senza miglioramento\n', ...
                    obj.episodeSinceLastImprovement, obj.maxEpisodesWithoutImprovement);
            end
            fprintf('═══════════════════════════════════════\n\n');
        end

        function delete(obj)
            if ~isempty(obj.logFile) && obj.logFile > 0
                fclose(obj.logFile);
            end
        end
    end

    methods (Access = private)
        % ----------------------------------------------------------------
        % Estrae una riga di valori scalari da una struct risultato.
        % fields è un cell array di nomi campo; ritorna vettore numerico.
        % ----------------------------------------------------------------
        function row = extractMetricRow(obj, result, fields) %#ok<INUSL>
            row = zeros(1, length(fields));
            for fi = 1:length(fields)
                f = fields{fi};
                if isfield(result, f)
                    row(fi) = result.(f);
                end
            end
        end
    end
end

% Helper function (scope file, fuori dalla classe)
function result = iif(condition, trueVal, falseVal)
    if condition, result = trueVal; else, result = falseVal; end
end