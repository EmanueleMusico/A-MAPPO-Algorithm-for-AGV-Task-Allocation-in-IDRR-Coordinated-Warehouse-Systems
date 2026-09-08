classdef MAPPO_Agent < handle
    % MAPPO_AGENT - Multi-Agent Proximal Policy Optimization (VERSIONE CORRETTA FINALE)
    % Implementazione corretta con:
    % 1. GAE separato per ogni AGV
    % 2. Aggregazione in batch centrale
    % 3. Update multipli sul batch aggregato
    % 4. Gradienti mediati a livello di batch
    
    properties (Access = public)
        % NEURAL NETWORKS
        actorNetwork
        criticNetwork
        
        % ADAM OPTIMIZER STATE (CORRETTO)
        actorAdamState
        criticAdamState
        adamIteration = 0;

        deterministicMode = false;  % Flag per modalità deterministica (argmax)
        
        % TRAINING HYPERPARAMETERS
        learningRate = 3e-4;
        epsilon = 0.2;
        gamma = 0.99;
        lambda = 0.95;
        valueCoeff = 0.5;
        entropyCoeff = 0.01;
        
        adamBeta1 = 0.9;
        adamBeta2 = 0.999;
        adamEpsilon = 1e-8;
        
        maxGradNorm = 0.5;
        miniBatchSize = 64;
        ppoEpochs = 15;
        
        % MULTI-AGENT MANAGEMENT
        numAGVs
        agvExperienceBuffers
        agvGAEBuffers
        maxExperiencesPerAGV = 1000;
        
        % POLICY STORAGE
        oldActorNetwork
        
        % TRAINING STATE
        trainingStep = 0;
        totalExperiencesProcessed = 0;
        
        % MONITORING
        trainingMetrics
        attentionWeightsHistory
        verboseLogging = true;
        debugMode = false;
        
        % DEVICE
        useGPU = false;
    end
    
    properties (Access = public, Dependent)
        PolicyLoss
        ValueLoss
        EntropyLoss
        TotalLoss
    end
    
    methods
        function obj = MAPPO_Agent(numAGVs, varargin)
            % CONSTRUCTOR
            p = inputParser;
            addRequired(p, 'numAGVs', @(x) isscalar(x) && x > 0);
            addParameter(p, 'learningRate', 3e-4, @(x) isscalar(x) && x > 0);
            addParameter(p, 'epsilon', 0.2, @(x) isscalar(x) && x > 0);
            addParameter(p, 'gamma', 0.99, @(x) isscalar(x) && x > 0 && x <= 1);
            addParameter(p, 'lambda', 0.95, @(x) isscalar(x) && x >= 0 && x <= 1);
            addParameter(p, 'valueCoeff', 0.5, @(x) isscalar(x) && x >= 0);
            addParameter(p, 'entropyCoeff', 0.01, @(x) isscalar(x) && x >= 0);
            addParameter(p, 'maxGradNorm', 0.5, @(x) isscalar(x) && x > 0);
            addParameter(p, 'verboseLogging', true, @islogical);
            addParameter(p, 'useGPU', false, @islogical);
            addParameter(p, 'debugMode', false, @islogical);
            
            parse(p, numAGVs, varargin{:});
            
            obj.numAGVs = p.Results.numAGVs;
            obj.learningRate = p.Results.learningRate;
            obj.epsilon = p.Results.epsilon;
            obj.gamma = p.Results.gamma;
            obj.lambda = p.Results.lambda;
            obj.valueCoeff = p.Results.valueCoeff;
            obj.entropyCoeff = p.Results.entropyCoeff;
            obj.maxGradNorm = p.Results.maxGradNorm;
            obj.verboseLogging = p.Results.verboseLogging;
            obj.useGPU = p.Results.useGPU && canUseGPU();
            obj.debugMode = p.Results.debugMode;
            
            if obj.useGPU && ~canUseGPU()
                warning('GPU non disponibile, uso CPU');
                obj.useGPU = false;
            end
            
            obj.initializeTrainingMetrics();
            obj.initializeNetworks();
            obj.initializePerAGVBuffers();
            
            if obj.verboseLogging
                %fprintf('✅ MAPPO Agent inizializzato: %d AGV, device=%s\n', ...
                %    obj.numAGVs, obj.getDeviceString());
            end
        end
        
        function [actionIndex, actionProbs, stateValue] = selectAction(obj, localFeatures, globalFeatures, actionMask)
            % SELEZIONE AZIONE
            if nargin < 4
                actionMask = true(5, 1);
            end
            
            try
                localTensor = dlarray(single(localFeatures(:)), 'CB');
                globalTensor = dlarray(single(globalFeatures(:)), 'CB');
                
                if obj.useGPU
                    localTensor = gpuArray(localTensor);
                    globalTensor = gpuArray(globalTensor);
                end
                
                actionLogits = predict(obj.actorNetwork, localTensor);
                actionLogits = actionLogits - max(actionLogits);
                
                actionMask = logical(actionMask(:));
                if ~all(actionMask)
                    actionLogits(~actionMask) = -Inf;
                end
                
                actionProbs = softmax(actionLogits);
                stateValue = predict(obj.criticNetwork, globalTensor);
                
                if obj.useGPU
                    actionProbs = gather(actionProbs);
                    stateValue = gather(stateValue);
                end
                
                actionProbs = double(extractdata(actionProbs));
                stateValue = double(extractdata(stateValue));
                
                actionProbs = actionProbs(:);
                actionProbs = actionProbs / sum(actionProbs);
                
                actionIndex = randsample(1:5, 1, true, actionProbs);
                
                if obj.debugMode
                    obj.saveAttentionWeights(actionProbs);
                end
                
            catch ME
                fprintf('⚠️ Errore selectAction: %s\n', ME.message);
                actionIndex = randi(5);
                actionProbs = ones(5,1) / 5;
                stateValue = 0;
            end
        end

        function [actionIndex, actionProbs, stateValue] = selectActionDeterministic(obj, localFeatures, globalFeatures, actionMask)
            % SELEZIONE AZIONE DETERMINISTICA (ARGMAX)
            % Usato durante testing per valutazione policy senza stocasticità
            % Seleziona azione con probabilità massima invece di sampling

            if nargin < 4
                actionMask = true(5, 1);
            end

            try
                % Converti feature in dlarray
                localTensor = dlarray(single(localFeatures(:)), 'CB');
                globalTensor = dlarray(single(globalFeatures(:)), 'CB');

                if obj.useGPU
                    localTensor = gpuArray(localTensor);
                    globalTensor = gpuArray(globalTensor);
                end

                % Forward pass actor network
                actionLogits = predict(obj.actorNetwork, localTensor);
                actionLogits = actionLogits - max(actionLogits);  % Numerical stability

                % Applica action mask
                actionMask = logical(actionMask(:));
                if ~all(actionMask)
                    actionLogits(~actionMask) = -Inf;
                end

                % Calcola probabilità azioni
                actionProbs = softmax(actionLogits);

                % Forward pass critic network
                stateValue = predict(obj.criticNetwork, globalTensor);

                % Converti a CPU se necessario
                if obj.useGPU
                    actionProbs = gather(actionProbs);
                    stateValue = gather(stateValue);
                end

                actionProbs = double(extractdata(actionProbs));
                stateValue = double(extractdata(stateValue));

                actionProbs = actionProbs(:);
                actionProbs = actionProbs / sum(actionProbs);  % Normalizza

                % ========================================
                % SELEZIONE DETERMINISTICA: ARGMAX
                % ========================================
                [~, actionIndex] = max(actionProbs);

                % Log se in debug mode
                if obj.debugMode
                    obj.saveAttentionWeights(actionProbs);
                    if obj.verboseLogging
                        fprintf('  [Deterministic] Action %d selected (p=%.4f)\n', ...
                            actionIndex, actionProbs(actionIndex));
                    end
                end

            catch ME
                fprintf('⚠️ Errore selectActionDeterministic: %s\n', ME.message);
                % Fallback: selezione random tra azioni valide
                validActions = find(actionMask);
                actionIndex = validActions(randi(length(validActions)));
                actionProbs = ones(5,1) / 5;
                stateValue = 0;
            end
        end

        function [actionIndex, actionProbs, stateValue] = selectActionAdaptive(obj, localFeatures, globalFeatures, actionMask)
            % SELEZIONE AZIONE ADATTIVA
            % Usa modalità deterministica o stocastica in base a obj.deterministicMode

            if obj.deterministicMode
                [actionIndex, actionProbs, stateValue] = obj.selectActionDeterministic(...
                    localFeatures, globalFeatures, actionMask);
            else
                [actionIndex, actionProbs, stateValue] = obj.selectAction(...
                    localFeatures, globalFeatures, actionMask);
            end
        end

        
        function storeExperience(obj, agvId, localFeatures, globalFeatures, actionIndex, ...
                reward, nextLocalFeatures, nextGlobalFeatures, done, actionMask)
            % MEMORIZZA ESPERIENZA PER AGV SPECIFICO
            if nargin < 10
                actionMask = true(5, 1);
            end
            
            try
                if agvId < 1 || agvId > obj.numAGVs
                    error('agvId fuori range: %d', agvId);
                end
                
                experience = struct(...
                    'localFeatures', localFeatures(:), ...
                    'globalFeatures', globalFeatures(:), ...
                    'actionIndex', actionIndex, ...
                    'reward', reward, ...
                    'nextLocalFeatures', nextLocalFeatures(:), ...
                    'nextGlobalFeatures', nextGlobalFeatures(:), ...
                    'done', done, ...
                    'timestamp', tic, ...
                    'actionMask', actionMask(:));
                
                if isempty(obj.agvExperienceBuffers{agvId})
                    obj.agvExperienceBuffers{agvId} = experience;
                else
                    obj.agvExperienceBuffers{agvId}(end+1) = experience;
                end
                
                if length(obj.agvExperienceBuffers{agvId}) > obj.maxExperiencesPerAGV
                    obj.agvExperienceBuffers{agvId}(1) = [];
                end
                
                obj.totalExperiencesProcessed = obj.totalExperiencesProcessed + 1;
                
            catch ME
                if obj.verboseLogging
                    fprintf('⚠️ Warning storeExperience AGV %d: %s\n', agvId, ME.message);
                end
            end
        end
        
        function success = trainStep(obj, minExperiences)
            % TRAINING STEP MAPPO CORRETTO
            % Procedura:
            % 1. Calcola GAE separatamente per ogni AGV
            % 2. Aggrega tutti i dati in un batch centrale
            % 3. Esegue update multipli sul batch aggregato
            % 4. Gradienti mediati a livello di batch
            
            if nargin < 2
                minExperiences = obj.miniBatchSize;
            end
            
            success = false;
            
            try
                % ========================================
                % FASE 1: CALCOLA GAE PER OGNI AGV
                % ========================================
                
                if obj.verboseLogging
                    fprintf('\n🔄 Training Step %d - Calcolo GAE per AGV...\n', obj.trainingStep + 1);
                end
                
                totalValidExperiences = 0;
                
                for agvId = 1:obj.numAGVs
                    buffer = obj.agvExperienceBuffers{agvId};
                    
                    if isempty(buffer)
                        obj.agvGAEBuffers{agvId} = [];
                        continue;
                    end
                    
                    % Calcola GAE per questo AGV
                    [advantages_i, returns_i, values_i] = obj.computeGAEForAGV(agvId, buffer);
                    
                    % Salva GAE buffer
                    obj.agvGAEBuffers{agvId}.advantages = advantages_i;
                    obj.agvGAEBuffers{agvId}.returns = returns_i;
                    obj.agvGAEBuffers{agvId}.values = values_i;
                    obj.agvGAEBuffers{agvId}.count = length(advantages_i);
                    
                    totalValidExperiences = totalValidExperiences + length(advantages_i);
                    
                    if obj.verboseLogging
                        fprintf('   AGV_%d: %d exp, mean_adv=%.3f\n', agvId, length(advantages_i), mean(advantages_i));
                    end
                end
                
                if totalValidExperiences < minExperiences
                    if obj.verboseLogging
                        fprintf('📊 Training saltato: %d/%d esperienze\n', ...
                            totalValidExperiences, minExperiences);
                    end
                    return;
                end
                
                % ========================================
                % FASE 2: AGGREGA DATI IN BATCH CENTRALE
                % ========================================
                
                if obj.verboseLogging
                    fprintf('📦 Aggregazione batch centrale da %d AGV...\n', obj.numAGVs);
                end
                
                [centralBatchStates, centralBatchGlobalStates, centralBatchActions, ...
                 centralBatchAdvantages, centralBatchReturns, centralBatchMasks] = ...
                    obj.aggregateCentralBatch();
                
                totalSamples = length(centralBatchStates);
                
                if totalSamples == 0
                    if obj.verboseLogging
                        fprintf('⚠️ Batch centrale vuoto\n');
                    end
                    return;
                end
                
                if obj.verboseLogging
                    fprintf('   Batch centrale: %d samples totali\n', totalSamples);
                end
                
                % ========================================
                % FASE 3: SALVA OLD POLICY PER PPO RATIO
                % ========================================
                
                obj.saveOldPolicy();
                
                % ========================================
                % FASE 4: TRAINING LOOP CON UPDATE MULTIPLI
                % ========================================
                
                if obj.verboseLogging
                    fprintf('🎯 Training con %d epoche PPO...\n', obj.ppoEpochs);
                end
                
                totalActorLoss = 0;
                totalCriticLoss = 0;
                totalEntropyLoss = 0;
                validGradientSteps = 0;
                
                for epoch = 1:obj.ppoEpochs
                    % Crea mini-batches dal batch centrale
                    batchIndices = obj.createMiniBatches(totalSamples);
                    
                    epochActorLoss = 0;
                    epochCriticLoss = 0;
                    epochEntropyLoss = 0;
                    epochSteps = 0;
                    
                    for batchIdx = 1:length(batchIndices)
                        indices = batchIndices{batchIdx};
                        
                        % Estrai mini-batch
                        batchStates = centralBatchStates(indices);
                        batchGlobalStates = centralBatchGlobalStates(indices);
                        batchActions = centralBatchActions(indices);
                        batchAdvantages = centralBatchAdvantages(indices);
                        batchReturns = centralBatchReturns(indices);
                        batchMasks = centralBatchMasks(indices);
                        
                        % Calcola gradienti e aggiorna (un update per mini-batch)
                        [actorLoss, criticLoss, entropyLoss, gradientValid] = ...
                            obj.computeAndApplyGradients(batchStates, batchGlobalStates, ...
                                batchActions, batchAdvantages, batchReturns, batchMasks);
                        
                        if gradientValid
                            epochActorLoss = epochActorLoss + actorLoss;
                            epochCriticLoss = epochCriticLoss + criticLoss;
                            epochEntropyLoss = epochEntropyLoss + entropyLoss;
                            epochSteps = epochSteps + 1;
                        end
                    end
                    
                    if epochSteps > 0
                        totalActorLoss = totalActorLoss + (epochActorLoss / epochSteps);
                        totalCriticLoss = totalCriticLoss + (epochCriticLoss / epochSteps);
                        totalEntropyLoss = totalEntropyLoss + (epochEntropyLoss / epochSteps);
                        validGradientSteps = validGradientSteps + 1;
                        
                        if obj.verboseLogging
                            fprintf('   Epoca %d/%d: Policy=%.4f, Value=%.4f, %d batches\n', ...
                                epoch, obj.ppoEpochs, epochActorLoss/epochSteps, ...
                                epochCriticLoss/epochSteps, epochSteps);
                        end
                    end
                end
                
                % ========================================
                % FASE 5: AGGIORNA METRICHE
                % ========================================
                
                if validGradientSteps > 0
                    avgActorLoss = totalActorLoss / validGradientSteps;
                    avgCriticLoss = totalCriticLoss / validGradientSteps;
                    avgEntropyLoss = totalEntropyLoss / validGradientSteps;
                    
                    obj.updateTrainingMetrics(avgActorLoss, avgCriticLoss, avgEntropyLoss);
                    obj.trainingStep = obj.trainingStep + 1;
                    
                    success = true;
                    
                    if obj.verboseLogging
                        fprintf('✅ Training completato: Policy=%.4f, Value=%.4f, Entropy=%.4f\n', ...
                            avgActorLoss, avgCriticLoss, avgEntropyLoss);
                        fprintf('   Total samples: %d, Updates: %d\n\n', totalSamples, validGradientSteps);
                    end
                else
                    if obj.verboseLogging
                        fprintf('⚠️ Training fallito: nessun gradiente valido\n\n');
                    end
                end
                
                % Pulisce buffer
                obj.cleanupCompletedEpisodes();
                
            catch ME
                fprintf('❌ Errore trainStep: %s\n', ME.message);
                if obj.debugMode
                    fprintf('Stack trace:\n%s\n', ME.getReport());
                end
            end
        end
        
        function [advantages, returns, values] = computeGAEForAGV(obj, agvId, buffer)
            % CALCOLA GAE PER SINGOLO AGV (Equazione 6 paper)
            numExperiences = length(buffer);
            advantages = zeros(numExperiences, 1);
            returns = zeros(numExperiences, 1);
            values = zeros(numExperiences, 1);
            
            if numExperiences == 0
                return;
            end
            
            try
                rewards = [buffer.reward];
                dones = [buffer.done];
                
                % Calcola V(s^i_t)
                for i = 1:numExperiences
                    globalTensor = dlarray(single(buffer(i).globalFeatures(:)), 'CB');
                    if obj.useGPU
                        globalTensor = gpuArray(globalTensor);
                    end
                    stateValue = predict(obj.criticNetwork, globalTensor);
                    values(i) = double(gather(extractdata(stateValue)));
                end
                
                % Calcola V(s^i_{t+1})
                nextValues = zeros(numExperiences, 1);
                for i = 1:numExperiences
                    if ~dones(i)
                        nextGlobalTensor = dlarray(single(buffer(i).nextGlobalFeatures(:)), 'CB');
                        if obj.useGPU
                            nextGlobalTensor = gpuArray(nextGlobalTensor);
                        end
                        nextStateValue = predict(obj.criticNetwork, nextGlobalTensor);
                        nextValues(i) = double(gather(extractdata(nextStateValue)));
                    else
                        nextValues(i) = 0;
                    end
                end
                
                % TD errors: δ^i_t
                tdErrors = rewards(:) + obj.gamma * nextValues - values;
                
                % GAE backwards
                gae = 0;
                for i = numExperiences:-1:1
                    if dones(i)
                        gae = tdErrors(i);
                    else
                        gae = tdErrors(i) + obj.gamma * obj.lambda * gae;
                    end
                    advantages(i) = gae;
                end
                
                % Returns
                returns = advantages + values;
                
                % Normalizza advantages
                if std(advantages) > 1e-8
                    advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8);
                end
                
            catch ME
                if obj.verboseLogging
                    fprintf('⚠️ Warning computeGAEForAGV AGV_%d: %s\n', agvId, ME.message);
                end
                advantages = zeros(numExperiences, 1);
                returns = rewards(:);
            end
        end
        
        function [centralBatchStates, centralBatchGlobalStates, centralBatchActions, ...
                centralBatchAdvantages, centralBatchReturns, centralBatchMasks] = aggregateCentralBatch(obj)
            % AGGREGA TUTTI I DATI DEGLI AGV IN UN BATCH CENTRALE
            % Dopo che GAE è stato calcolato separatamente per ogni AGV,
            % tutti i dati vengono unificati per il training centralizzato
            
            centralBatchStates = {};
            centralBatchGlobalStates = {};
            centralBatchActions = [];
            centralBatchAdvantages = [];
            centralBatchReturns = [];
            centralBatchMasks = {};
            
            for agvId = 1:obj.numAGVs
                buffer = obj.agvExperienceBuffers{agvId};
                gaeBuffer = obj.agvGAEBuffers{agvId};
                
                if isempty(buffer) || isempty(gaeBuffer) || gaeBuffer.count == 0
                    continue;
                end
                
                % Aggiungi dati di questo AGV al batch centrale
                centralBatchStates = [centralBatchStates, {buffer.localFeatures}];
                centralBatchGlobalStates = [centralBatchGlobalStates, {buffer.globalFeatures}];
                centralBatchActions = [centralBatchActions, [buffer.actionIndex]];
                centralBatchAdvantages = [centralBatchAdvantages; gaeBuffer.advantages];
                centralBatchReturns = [centralBatchReturns; gaeBuffer.returns];
                centralBatchMasks = [centralBatchMasks, {buffer.actionMask}];
            end
        end
        
        function [actorLoss, criticLoss, entropyLoss, gradientValid] = ...
                computeAndApplyGradients(obj, batchStates, batchGlobalStates, ...
                    batchActions, batchAdvantages, batchReturns, batchMasks)
            % CALCOLA E APPLICA GRADIENTI (un update per mini-batch)
            gradientValid = false;
            actorLoss = 0;
            criticLoss = 0;
            entropyLoss = 0;
            
            try
                batchSize = length(batchStates);
                
                % Prepara tensori
                stateMatrix = zeros(109, batchSize, 'single');
                globalStateMatrix = zeros(109, batchSize, 'single');
                maskMatrix = false(5, batchSize);
                
                for i = 1:batchSize
                    stateMatrix(:, i) = single(batchStates{i}(:));
                    globalStateMatrix(:, i) = single(batchGlobalStates{i}(:));
                    maskMatrix(:, i) = logical(batchMasks{i}(:));
                end
                
                stateTensor = dlarray(stateMatrix, 'CB');
                globalStateTensor = dlarray(globalStateMatrix, 'CB');
                actionsTensor = dlarray(single(batchActions(:)), 'CB');
                advantagesTensor = dlarray(single(batchAdvantages(:)), 'CB');
                returnsTensor = dlarray(single(batchReturns(:)), 'CB');
                
                if obj.useGPU
                    stateTensor = gpuArray(stateTensor);
                    globalStateTensor = gpuArray(globalStateTensor);
                    actionsTensor = gpuArray(actionsTensor);
                    advantagesTensor = gpuArray(advantagesTensor);
                    returnsTensor = gpuArray(returnsTensor);
                    maskMatrix = gpuArray(maskMatrix);
                end
                
                % ACTOR GRADIENTS
                [actorGrad, actorLoss, entropyLoss] = dlfeval(@obj.actorLossFunction, ...
                    obj.actorNetwork, obj.oldActorNetwork, stateTensor, ...
                    actionsTensor, advantagesTensor, maskMatrix);
                
                % Gradient clipping actor
                actorGradNorm = sqrt(sum(cellfun(@(g) sum(extractdata(g(:)).^2), ...
                    actorGrad.Value), 'all'));
                
                if actorGradNorm > obj.maxGradNorm
                    scaleFactor = obj.maxGradNorm / actorGradNorm;
                    actorGrad.Value = cellfun(@(g) g * scaleFactor, actorGrad.Value, 'UniformOutput', false);
                end
                
                % CRITIC GRADIENTS
                [criticGrad, criticLoss] = dlfeval(@obj.criticLossFunction, ...
                    obj.criticNetwork, globalStateTensor, returnsTensor);
                
                % Gradient clipping critic
                criticGradNorm = sqrt(sum(cellfun(@(g) sum(extractdata(g(:)).^2), ...
                    criticGrad.Value), 'all'));
                
                if criticGradNorm > obj.maxGradNorm
                    scaleFactor = obj.maxGradNorm / criticGradNorm;
                    criticGrad.Value = cellfun(@(g) g * scaleFactor, criticGrad.Value, 'UniformOutput', false);
                end
                
                % AGGIORNA CON ADAM
                obj.adamIteration = obj.adamIteration + 1;
                
                [obj.actorNetwork.Learnables, obj.actorAdamState] = ...
                    obj.customAdamUpdate(obj.actorNetwork.Learnables, actorGrad, ...
                        obj.actorAdamState, obj.adamIteration);
                
                [obj.criticNetwork.Learnables, obj.criticAdamState] = ...
                    obj.customAdamUpdate(obj.criticNetwork.Learnables, criticGrad, ...
                        obj.criticAdamState, obj.adamIteration);
                
                % Converti loss
                actorLoss = double(gather(extractdata(actorLoss)));
                criticLoss = double(gather(extractdata(criticLoss)));
                entropyLoss = double(gather(extractdata(entropyLoss)));
                
                gradientValid = true;
                
            catch ME
                if obj.verboseLogging
                    fprintf('⚠️ Warning computeAndApplyGradients: %s\n', ME.message);
                end
            end
        end
        
        function [updatedLearnables, updatedState] = customAdamUpdate(obj, learnables, ...
                gradients, adamState, iteration)
            % ADAM UPDATE CUSTOM
            
            if isempty(adamState) || ~isfield(adamState, 'm')
                adamState.m = cell(size(learnables, 1), 1);
                adamState.v = cell(size(learnables, 1), 1);
                for i = 1:size(learnables, 1)
                    paramSize = size(learnables.Value{i});
                    adamState.m{i} = zeros(paramSize, 'like', learnables.Value{i});
                    adamState.v{i} = zeros(paramSize, 'like', learnables.Value{i});
                end
            end
            
            biasCorrection1 = 1 - obj.adamBeta1^iteration;
            biasCorrection2 = 1 - obj.adamBeta2^iteration;
            
            updatedLearnables = learnables;
            updatedState = adamState;
            
            for i = 1:size(learnables, 1)
                grad = extractdata(gradients.Value{i});
                
                updatedState.m{i} = obj.adamBeta1 * adamState.m{i} + (1 - obj.adamBeta1) * grad;
                updatedState.v{i} = obj.adamBeta2 * adamState.v{i} + (1 - obj.adamBeta2) * (grad.^2);
                
                mHat = updatedState.m{i} / biasCorrection1;
                vHat = updatedState.v{i} / biasCorrection2;
                
                paramUpdate = obj.learningRate * mHat ./ (sqrt(vHat) + obj.adamEpsilon);
                updatedLearnables.Value{i} = learnables.Value{i} - paramUpdate;
            end
        end
        
        function [gradients, loss, entropyLoss] = actorLossFunction(obj, actorNet, ...
                oldActorNet, states, actions, advantages, masks)
            % ACTOR LOSS CON PPO CLIPPING E ACTION MASKING
            
            actionLogits = forward(actorNet, states);
            actionLogits = actionLogits - max(actionLogits, [], 1);
            
            for i = 1:size(states, 2)
                mask_i = masks(:, i);
                actionLogits(~mask_i, i) = -Inf;
            end
            
            actionProbs = softmax(actionLogits);
            
            oldActionLogits = forward(oldActorNet, states);
            oldActionLogits = oldActionLogits - max(oldActionLogits, [], 1);
            
            for i = 1:size(states, 2)
                mask_i = masks(:, i);
                oldActionLogits(~mask_i, i) = -Inf;
            end
            
            oldActionProbs = softmax(oldActionLogits);
            
            batchSize = size(states, 2);
            actionsData = double(gather(extractdata(actions(:))));
            
            selectedProbs = dlarray(zeros(batchSize, 1));
            oldSelectedProbs = dlarray(zeros(batchSize, 1));
            
            for i = 1:batchSize
                actionIdx = round(actionsData(i));
                actionIdx = max(1, min(5, actionIdx));
                selectedProbs(i) = actionProbs(actionIdx, i);
                oldSelectedProbs(i) = oldActionProbs(actionIdx, i);
            end
            
            logProbs = log(selectedProbs + 1e-8);
            oldLogProbs = log(oldSelectedProbs + 1e-8);
            ratio = exp(logProbs - oldLogProbs);
            clippedRatio = max(min(ratio, 1 + obj.epsilon), 1 - obj.epsilon);
            
            policyLoss1 = ratio .* advantages;
            policyLoss2 = clippedRatio .* advantages;
            loss = -mean(min(policyLoss1, policyLoss2));
            
            entropy = -sum(actionProbs .* log(actionProbs + 1e-8), 1);
            entropyLoss = -mean(entropy);
            
            gradients = dlgradient(loss, actorNet.Learnables);
        end
        
        function [gradients, loss] = criticLossFunction(obj, criticNet, states, returns)
            % CRITIC LOSS MSE
            values = forward(criticNet, states);
            values = squeeze(values);
            numExperiences = size(states, 2);
            values = reshape(values, [numExperiences, 1]);
            squaredErrors = (values - returns).^2;
            loss = mean(squaredErrors);
            gradients = dlgradient(loss, criticNet.Learnables);
        end
        
        function saveOldPolicy(obj)
            % SALVA POLICY CORRENTE PER PPO RATIO
            if isempty(obj.oldActorNetwork)
                obj.oldActorNetwork = obj.actorNetwork;
            else
                obj.oldActorNetwork.Learnables = obj.actorNetwork.Learnables;
            end
        end
        
        function batchIndices = createMiniBatches(obj, numExperiences)
            % CREA MINI-BATCHES RANDOMIZZATI
            batchIndices = {};
            
            if numExperiences < obj.miniBatchSize
                batchIndices{1} = 1:numExperiences;
                return;
            end
            
            indices = randperm(numExperiences);
            numBatches = floor(numExperiences / obj.miniBatchSize);
            
            for i = 1:numBatches
                startIdx = (i-1) * obj.miniBatchSize + 1;
                endIdx = i * obj.miniBatchSize;
                batchIndices{i} = indices(startIdx:endIdx);
            end
            
            if numBatches * obj.miniBatchSize < numExperiences
                batchIndices{end+1} = indices((numBatches * obj.miniBatchSize + 1):end);
            end
        end
        
        function clearBuffers(obj)
            % PULISCE TUTTI I BUFFER
            for agvId = 1:obj.numAGVs
                obj.agvExperienceBuffers{agvId} = [];
                obj.agvGAEBuffers{agvId} = [];
            end
            
            if obj.verboseLogging
                fprintf('🧹 Buffer puliti per tutti gli AGV\n');
            end
        end
        
        function updateTrainingMetrics(obj, actorLoss, criticLoss, entropyLoss)
            % AGGIORNA METRICHE TRAINING
            obj.trainingMetrics.policyLossHistory(end+1) = actorLoss;
            obj.trainingMetrics.valueLossHistory(end+1) = criticLoss;
            obj.trainingMetrics.entropyLossHistory(end+1) = entropyLoss;
            
            totalLoss = actorLoss + obj.valueCoeff * criticLoss - obj.entropyCoeff * entropyLoss;
            obj.trainingMetrics.totalLossHistory(end+1) = totalLoss;
            obj.trainingMetrics.trainingSteps = obj.trainingMetrics.trainingSteps + 1;
        end
        
        function cleanupCompletedEpisodes(obj)
            % PULISCE BUFFER DOPO TRAINING
            for agvId = 1:obj.numAGVs
                obj.agvExperienceBuffers{agvId} = [];
                obj.agvGAEBuffers{agvId} = [];
            end
        end
        
        function saveAttentionWeights(obj, actionProbs)
            % SALVA ATTENTION WEIGHTS PER DEBUG
            if obj.debugMode
                if isempty(obj.attentionWeightsHistory)
                    obj.attentionWeightsHistory = actionProbs';
                else
                    obj.attentionWeightsHistory(end+1, :) = actionProbs';
                end
                
                if size(obj.attentionWeightsHistory, 1) > 1000
                    obj.attentionWeightsHistory(1, :) = [];
                end
            end
        end
        
        function deviceStr = getDeviceString(obj)
            % RESTITUISCE STRINGA DEVICE
            if obj.useGPU
                deviceStr = 'GPU';
            else
                deviceStr = 'CPU';
            end
        end
        
        function resetAgent(obj)
            % RESET COMPLETO AGENTE
            for agvId = 1:obj.numAGVs
                obj.agvExperienceBuffers{agvId} = [];
                obj.agvGAEBuffers{agvId} = [];
            end
            
            obj.initializeTrainingMetrics();
            obj.trainingStep = 0;
            obj.totalExperiencesProcessed = 0;
            obj.adamIteration = 0;
            
            if obj.verboseLogging
                fprintf('🔄 MAPPO Agent reset completato\n');
            end
        end
        
        function resetAGVBuffers(obj, agvIds)
            % RESET BUFFER PER AGV SPECIFICI
            if nargin < 2
                agvIds = 1:obj.numAGVs;
            end
            
            for agvId = agvIds
                if agvId >= 1 && agvId <= obj.numAGVs
                    obj.agvExperienceBuffers{agvId} = [];
                    obj.agvGAEBuffers{agvId} = [];
                end
            end
            
            if obj.verboseLogging
                fprintf('🧹 Buffer reset per AGV: %s\n', mat2str(agvIds));
            end
        end
        
        function saveAgent(obj, filename)
            % SALVA AGENTE SU FILE
            try
                actorNet = obj.actorNetwork;
                criticNet = obj.criticNetwork;
                hyperparams = struct(...
                    'learningRate', obj.learningRate, ...
                    'epsilon', obj.epsilon, ...
                    'gamma', obj.gamma, ...
                    'lambda', obj.lambda, ...
                    'valueCoeff', obj.valueCoeff, ...
                    'entropyCoeff', obj.entropyCoeff);
                metrics = obj.trainingMetrics;
                trainingStep = obj.trainingStep;
                
                save(filename, 'actorNet', 'criticNet', 'hyperparams', 'metrics', 'trainingStep');
                fprintf('✅ Agente salvato: %s\n', filename);
            catch ME
                fprintf('❌ Errore salvataggio: %s\n', ME.message);
            end
        end
        
        function loadAgent(obj, filename)
            % CARICA AGENTE DA FILE
            try
                data = load(filename);
                obj.actorNetwork = data.actorNet;
                obj.criticNetwork = data.criticNet;
                
                if isfield(data, 'hyperparams')
                    obj.learningRate = data.hyperparams.learningRate;
                    obj.epsilon = data.hyperparams.epsilon;
                    obj.gamma = data.hyperparams.gamma;
                    obj.lambda = data.hyperparams.lambda;
                    obj.valueCoeff = data.hyperparams.valueCoeff;
                    obj.entropyCoeff = data.hyperparams.entropyCoeff;
                end
                
                if isfield(data, 'metrics')
                    obj.trainingMetrics = data.metrics;
                end
                
                if isfield(data, 'trainingStep')
                    obj.trainingStep = data.trainingStep;
                end
                
                obj.initializeAdamStates();
                
                fprintf('✅ Agente caricato: %s (step %d)\n', filename, obj.trainingStep);
            catch ME
                fprintf('❌ Errore caricamento: %s\n', ME.message);
            end
        end
        
        function plotTrainingMetrics(obj)
            % VISUALIZZA METRICHE TRAINING
            figure('Name', 'MAPPO Training Metrics', 'Position', [100, 100, 1200, 800]);
            
            subplot(2, 2, 1);
            plot(obj.trainingMetrics.policyLossHistory, 'b-', 'LineWidth', 1.5);
            xlabel('Training Step');
            ylabel('Policy Loss');
            title('Actor Policy Loss');
            grid on;
            
            subplot(2, 2, 2);
            plot(obj.trainingMetrics.valueLossHistory, 'r-', 'LineWidth', 1.5);
            xlabel('Training Step');
            ylabel('Value Loss');
            title('Critic Value Loss');
            grid on;
            
            subplot(2, 2, 3);
            plot(obj.trainingMetrics.entropyLossHistory, 'g-', 'LineWidth', 1.5);
            xlabel('Training Step');
            ylabel('Entropy Loss');
            title('Entropy Loss (Exploration)');
            grid on;
            
            subplot(2, 2, 4);
            plot(obj.trainingMetrics.totalLossHistory, 'k-', 'LineWidth', 1.5);
            xlabel('Training Step');
            ylabel('Total Loss');
            title('Total Combined Loss');
            grid on;
            
            sgtitle(sprintf('MAPPO Training Progress - %d Steps', obj.trainingStep));
        end
        
        function analyzeAttentionPattern(obj)
            % ANALIZZA PATTERN ATTENTION
            if isempty(obj.attentionWeightsHistory)
                fprintf('⚠️ Nessun dato attention disponibile\n');
                return;
            end
            
            meanWeights = mean(obj.attentionWeightsHistory, 1);
            stdWeights = std(obj.attentionWeightsHistory, 0, 1);
            
            fprintf('\n=== ANALISI ATTENTION PATTERN ===\n');
            fprintf('Campioni raccolti: %d\n', size(obj.attentionWeightsHistory, 1));
            fprintf('\nPeso medio attention per posizione:\n');
            for pos = 1:length(meanWeights)
                fprintf('  Posizione %d: %.3f ± %.3f\n', pos, meanWeights(pos), stdWeights(pos));
            end
            
            expectedUniform = 0.2;
            bias = abs(meanWeights - expectedUniform);
            maxBias = max(bias);
            
            fprintf('\nBias massimo da distribuzione uniforme: %.3f\n', maxBias);
            if maxBias > 0.1
                fprintf('⚠️ Possibile bias posizionale rilevato!\n');
            else
                fprintf('✅ Pattern attention bilanciato\n');
            end
            
            figure('Name', 'Attention Analysis');
            bar(meanWeights);
            hold on;
            errorbar(1:5, meanWeights, stdWeights, 'k.', 'LineWidth', 1.5);
            yline(expectedUniform, 'r--', 'Uniform', 'LineWidth', 2);
            xlabel('Posizione Task');
            ylabel('Probabilità Media');
            title('Distribuzione Attention Weights');
            legend('Mean', 'Std Dev', 'Uniform Expected');
            grid on;
        end
        
        function metrics = getTrainingMetrics(obj)
            % RESTITUISCE METRICHE TRAINING
            metrics = obj.trainingMetrics;
        end
    end
    
    methods (Access = private)
        % METODI PRIVATI DI INIZIALIZZAZIONE
        
        function initializeNetworks(obj)
            % INIZIALIZZA RETI ACTOR E CRITIC CON DIMENSIONI CORRETTE
            % Actor e Critic ricevono entrambi 109 feature (design modificato)
            % Le feature includono tutti gli embedding necessari per cooperazione

            try
                % ========== ACTOR NETWORK: 109 → 5 ==========
                % Input: 109 feature condivise (posizione AGV, congestione globale,
                %        task embeddings, overlap embeddings, metriche sistema)
                % Output: 5 logits per i 5 task candidati

                actorLayers = [
                    featureInputLayer(109, 'Name', 'actor_input', ...
                    'Normalization', 'none')

                    % Layer 1: Embedding generale delle feature
                    fullyConnectedLayer(128, 'Name', 'actor_embed1')
                    reluLayer('Name', 'actor_relu1')

                    % Layer 2: Feature extraction profonda
                    fullyConnectedLayer(256, 'Name', 'actor_extract')
                    reluLayer('Name', 'actor_relu2')

                    % Layer 3: Compression con attention implicita
                    fullyConnectedLayer(128, 'Name', 'actor_compress')
                    reluLayer('Name', 'actor_relu3')

                    % Layer 4: Attention mechanism preparation
                    % Questo layer impara a "prestare attenzione" alle feature rilevanti
                    fullyConnectedLayer(64, 'Name', 'actor_attention')
                    reluLayer('Name', 'actor_relu4')

                    % Layer 5: Action logits (senza attivazione - softmax applicato dopo)
                    fullyConnectedLayer(5, 'Name', 'actor_logits')
                    ];

                obj.actorNetwork = dlnetwork(layerGraph(actorLayers));


                % ========== CRITIC NETWORK: 109 → 1 ==========
                % Input: 109 feature condivise (identiche all'actor)
                % Output: 1 valore scalare (stima V(s))

                criticLayers = [
                    featureInputLayer(109, 'Name', 'critic_input', ...
                    'Normalization', 'none')

                    % Layer 1: Embedding generale delle feature
                    fullyConnectedLayer(128, 'Name', 'critic_embed1')
                    reluLayer('Name', 'critic_relu1')

                    % Layer 2: Feature extraction profonda
                    fullyConnectedLayer(256, 'Name', 'critic_extract')
                    reluLayer('Name', 'critic_relu2')

                    % Layer 3: Compression per value estimation
                    fullyConnectedLayer(128, 'Name', 'critic_compress')
                    reluLayer('Name', 'critic_relu3')

                    % Layer 4: Value preparation
                    fullyConnectedLayer(64, 'Name', 'critic_value_prep')
                    reluLayer('Name', 'critic_relu4')

                    % Layer 5: State value output (scalare, no activation)
                    fullyConnectedLayer(1, 'Name', 'critic_value')
                    ];

                obj.criticNetwork = dlnetwork(layerGraph(criticLayers));

            catch ME
                error('❌ Errore inizializzazione reti: %s\n   Stack: %s', ...
                    ME.message, ME.stack(1).name);
            end
        end
        
        function initializeAdamStates(obj)
            % INIZIALIZZA STATI ADAM (VERSIONE CORRETTA)
            obj.actorAdamState = struct();
            obj.actorAdamState.m = cell(size(obj.actorNetwork.Learnables, 1), 1);
            obj.actorAdamState.v = cell(size(obj.actorNetwork.Learnables, 1), 1);
            
            for i = 1:size(obj.actorNetwork.Learnables, 1)
                paramSize = size(obj.actorNetwork.Learnables.Value{i});
                obj.actorAdamState.m{i} = zeros(paramSize, 'like', obj.actorNetwork.Learnables.Value{i});
                obj.actorAdamState.v{i} = zeros(paramSize, 'like', obj.actorNetwork.Learnables.Value{i});
            end
            
            obj.criticAdamState = struct();
            obj.criticAdamState.m = cell(size(obj.criticNetwork.Learnables, 1), 1);
            obj.criticAdamState.v = cell(size(obj.criticNetwork.Learnables, 1), 1);
            
            for i = 1:size(obj.criticNetwork.Learnables, 1)
                paramSize = size(obj.criticNetwork.Learnables.Value{i});
                obj.criticAdamState.m{i} = zeros(paramSize, 'like', obj.criticNetwork.Learnables.Value{i});
                obj.criticAdamState.v{i} = zeros(paramSize, 'like', obj.criticNetwork.Learnables.Value{i});
            end
            
            obj.adamIteration = 0;
            
            if obj.verboseLogging
                fprintf('✅ Stati Adam inizializzati correttamente\n');
            end
        end
        
        function initializeTrainingMetrics(obj)
            % INIZIALIZZA STRUCT METRICHE TRAINING
            obj.trainingMetrics = struct(...
                'policyLossHistory', [], ...
                'valueLossHistory', [], ...
                'entropyLossHistory', [], ...
                'totalLossHistory', [], ...
                'trainingSteps', 0);
        end
        
        function initializePerAGVBuffers(obj)
            % INIZIALIZZA BUFFER PER OGNI AGV
            obj.agvExperienceBuffers = cell(obj.numAGVs, 1);
            obj.agvGAEBuffers = cell(obj.numAGVs, 1);
            
            for i = 1:obj.numAGVs
                obj.agvExperienceBuffers{i} = [];
                obj.agvGAEBuffers{i} = [];
            end
        end
    end
    
    % DEPENDENT PROPERTIES GETTERS
    methods
        function loss = get.PolicyLoss(obj)
            if isfield(obj.trainingMetrics, 'policyLossHistory') && ...
               ~isempty(obj.trainingMetrics.policyLossHistory)
                loss = obj.trainingMetrics.policyLossHistory(end);
            else
                loss = NaN;
            end
        end
        
        function loss = get.ValueLoss(obj)
            if isfield(obj.trainingMetrics, 'valueLossHistory') && ...
               ~isempty(obj.trainingMetrics.valueLossHistory)
                loss = obj.trainingMetrics.valueLossHistory(end);
            else
                loss = NaN;
            end
        end
        
        function loss = get.EntropyLoss(obj)
            if isfield(obj.trainingMetrics, 'entropyLossHistory') && ...
               ~isempty(obj.trainingMetrics.entropyLossHistory)
                loss = obj.trainingMetrics.entropyLossHistory(end);
            else
                loss = NaN;
            end
        end
        
        function loss = get.TotalLoss(obj)
            policyLoss = obj.PolicyLoss;
            valueLoss = obj.ValueLoss;
            entropyLoss = obj.EntropyLoss;
            
            if ~isnan(policyLoss) && ~isnan(valueLoss) && ~isnan(entropyLoss)
                loss = policyLoss + obj.valueCoeff * valueLoss - obj.entropyCoeff * entropyLoss;
            else
                loss = NaN;
            end
        end
    end
end