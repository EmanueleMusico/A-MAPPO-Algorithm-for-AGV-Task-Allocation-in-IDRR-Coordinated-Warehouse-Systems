%% MAIN TRAINING - MAPPO TASK ALLOCATION MULTI-AGV
% Avvia training MAPPO tramite Training_Manager.
% Il test intermedio (ogni testInterval episodi) esegue:
%   numTestEpisodes=10 episodi da 3600s ciascuno, FIFO vs RL deterministico.
% La simulazione di test si ferma per timeout (non per completamento task).

clear; close all; clc;

fprintf('═══════════════════════════════════════════════════\n');
fprintf('TRAINING MAPPO PER TASK ALLOCATION MULTI-AGV\n');
fprintf('═══════════════════════════════════════════════════\n\n');

%% CONFIGURAZIONE
CONFIG = struct();
CONFIG.nAGV                = 7;
CONFIG.maxTasks            = 21;    % Task per episodio training
CONFIG.maxTime             = 3600;  % Timeout episodio training [s]
CONFIG.taskGenRate         = 0.05;   % Rate Poisson [task/s]
CONFIG.preGeneratedTasks   = 10;    % Task generati a t=0 con generateRandomTask()

CONFIG.numEpisodes           = 5000;
CONFIG.episodesPerTraining   = 100;
CONFIG.testInterval          = 200;
CONFIG.checkpointInterval    = 1000;

CONFIG.numTestEpisodes       = 10;   % Episodi per test session
CONFIG.numTestTasks          = 200;  % Task per episodio test (per riempire 3600s)

CONFIG.experimentName = sprintf('MAPPO_AGV%d_T%d_%s', ...
    CONFIG.nAGV, CONFIG.maxTasks, datestr(now, 'yyyymmdd_HHMMSS'));

fprintf('CONFIGURAZIONE:\n');
fprintf('  AGV: %d | maxTasks(training): %d | maxTime: %ds\n', ...
    CONFIG.nAGV, CONFIG.maxTasks, CONFIG.maxTime);
fprintf('  preGeneratedTasks: %d | taskGenRate: %.2f task/s\n', ...
    CONFIG.preGeneratedTasks, CONFIG.taskGenRate);
fprintf('  Episodi totali: %d | EpPerTraining: %d\n', ...
    CONFIG.numEpisodes, CONFIG.episodesPerTraining);
fprintf('  Test ogni %d ep: %d ep x %ds (numTestTasks=%d)\n', ...
    CONFIG.testInterval, CONFIG.numTestEpisodes, CONFIG.maxTime, CONFIG.numTestTasks);
fprintf('  Experiment: %s\n\n', CONFIG.experimentName);

%% CREAZIONE TRAINING MANAGER
fprintf('Inizializzazione Training Manager...\n');

try
    trainingManager = Training_Manager(CONFIG.nAGV, CONFIG.maxTasks, ...
        CONFIG.maxTime, CONFIG.taskGenRate, ...
        'preGeneratedTasks',   CONFIG.preGeneratedTasks, ...
        'episodesPerTraining', CONFIG.episodesPerTraining, ...
        'experimentName',      CONFIG.experimentName, ...
        'verboseLogging',      true);

    % Parametri test session
    trainingManager.testEpisodesInterval = CONFIG.testInterval;
    trainingManager.checkpointInterval   = CONFIG.checkpointInterval;
    trainingManager.numTestEpisodes      = CONFIG.numTestEpisodes;  % 10
    trainingManager.numTestTasks         = CONFIG.numTestTasks;     % 200

    % Parametri early stopping
    trainingManager.earlyStoppingEnabled          = true;
    trainingManager.significanceLevel             = 0.05;
    trainingManager.maxEpisodesWithoutImprovement = 1000;

    fprintf('Training Manager inizializzato (savePath: %s)\n\n', trainingManager.savePath);

catch ME
    fprintf('ERRORE inizializzazione: %s\n', ME.message);
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    return;
end

%% ESECUZIONE TRAINING
fprintf('Avvio training...\n');
fprintf('═══════════════════════════════════════════════════\n\n');

try
    tic;
    trainingManager.runTraining(CONFIG.numEpisodes);
    trainingTime = toc;

    fprintf('\n═══════════════════════════════════════════════════\n');
    fprintf('TRAINING COMPLETATO!\n');
    fprintf('  Tempo totale:      %.2f ore\n', trainingTime / 3600);
    fprintf('  Episodi eseguiti:  %d\n', trainingManager.currentEpisode);
    fprintf('  Training steps:    %d\n', trainingManager.totalTrainingSteps);
    fprintf('═══════════════════════════════════════════════════\n\n');

catch ME
    fprintf('\nERRORE durante training: %s\n', ME.message);
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end

    fprintf('\nSalvataggio risultati parziali...\n');
    try
        trainingManager.saveCheckpoint(trainingManager.currentEpisode);
        trainingManager.saveFinalResults();
        fprintf('Risultati parziali salvati\n');
    catch
        fprintf('Impossibile salvare risultati parziali\n');
    end
    return;
end

%% ANALISI RISULTATI FINALI
fprintf('ANALISI RISULTATI FINALI\n');
fprintf('───────────────────────────────────────────────────\n');

% Metriche training
if ~isempty(trainingManager.trainingMetrics.episodeRewards)
    rewards = trainingManager.trainingMetrics.episodeRewards;
    w       = min(50, length(rewards));
    fprintf('\nREWARD:\n');
    fprintf('  Finale:   %.2f\n', rewards(end));
    fprintf('  Medio:    %.2f\n', mean(rewards));
    fprintf('  Massimo:  %.2f\n', max(rewards));
    fprintf('  MA(ult %d): %.2f +/- %.2f\n', w, mean(rewards(end-w+1:end)), std(rewards(end-w+1:end)));
end

% Metriche testing (solo RL det vs IDRR)
if ~isempty(trainingManager.testingMetrics.testEpisodes)
    idrrFinal  = trainingManager.testingMetrics.idrr.productivity(end);
    rlDetFinal = trainingManager.testingMetrics.rlDeterministic.productivity(end);

    fprintf('\nTEST PERFORMANCE (ultimo test):\n');
    fprintf('  IDRR Baseline:    %.2f task/h\n', idrrFinal);
    fprintf('  RL Deterministic: %.2f task/h (%+.2f%%)\n', ...
        rlDetFinal, (rlDetFinal - idrrFinal) / idrrFinal * 100);
    fprintf('  Best RL Det:      %.2f task/h\n', trainingManager.bestTestPerformance);

    if trainingManager.convergenceAchieved
        fprintf('\n  CONVERGENZA RAGGIUNTA (early stopping)\n');
    else
        fprintf('\n  Convergenza non raggiunta (terminato per limite episodi)\n');
    end
end

% Loss finali
if ~isempty(trainingManager.trainingMetrics.policyLoss)
    fprintf('\nTRAINING LOSSES:\n');
    fprintf('  Policy Loss: %.4f -> %.4f\n', ...
        trainingManager.trainingMetrics.policyLoss(1), ...
        trainingManager.trainingMetrics.policyLoss(end));
    fprintf('  Value Loss:  %.4f -> %.4f\n', ...
        trainingManager.trainingMetrics.valueLoss(1), ...
        trainingManager.trainingMetrics.valueLoss(end));
end

fprintf('\n───────────────────────────────────────────────────\n');

%% INFO SALVATAGGIO
fprintf('\nRISULTATI SALVATI IN: %s\n', trainingManager.savePath);
fprintf('  final_results.mat  (trainingMetrics + testingMetrics + config)\n');
fprintf('  final_model.mat    (modello finale)\n');
fprintf('  best_model.mat     (modello con miglior test RL det)\n');
fprintf('  training_metrics.png/.fig (grafici)\n');
fprintf('  training.log       (log completo)\n');
fprintf('  checkpoint_ep*.mat (checkpoints periodici)\n');

fprintf('\n═══════════════════════════════════════════════════\n');
fprintf('DONE!\n');
fprintf('═══════════════════════════════════════════════════\n\n');