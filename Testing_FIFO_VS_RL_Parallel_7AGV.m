%% Testing_TABLE2_Par.m
% Versione parallela di Testing_TABLE2.m per 100 run FIFO vs MAPPO.
%
% DIFFERENZE rispetto alla versione seriale:
%   - Task arrays generati serialmente PRIMA del parfor (riproducibilità garantita)
%   - Tutti gli oggetti handle (agvSystem, agvEnvironment, mappoAgent) creati
%     localmente dentro parfor (vincolo MATLAB su classi handle)
%   - Modello RL caricato da disco in ogni worker
%   - avgResolvingTime calcolato da agvStateTimes.Resolving (corretto dopo fix classe)
%   - Risultati raccolti in cell arrays, aggregati dopo il parfor
%
% PREREQUISITO: AGV_System_IDRR_RL.m con fix su executeRes1 e completeManeuver.

clear; clc;

%% ---- PARAMETRI (identici allo script seriale) ----
nAGV               = 7;
maxTime            = 3600;
maxTasks           = inf;
numTasksToGenerate = 1000;
numRuns            = 100;
taskGenRate        = 0;
modelPath          = 'best_model_C3.mat';

fprintf('================================================\n');
fprintf('TESTING TABLE2 PARALLELO (%d run)\n', numRuns);
fprintf('================================================\n\n');

%% ---- PRE-GENERAZIONE TASK (seriale, garantisce riproducibilità) ----
% Ogni run usa rng(7345+run) - identico allo script originale.
% Richiedono agvSystem solo per leggere pickupNodes/dropoffNodes:
% questi non dipendono dallo stato dinamico, basta un'istanza temporanea.
fprintf('Pre-generazione task per tutte le run...\n');
tmpSys     = AGV_System_IDRR_RL(nAGV, maxTime, maxTasks, taskGenRate, ...
    'enableVisualization', false, 'operatingMode', 'IDRR', ...
    'verboseLogging', false, 'debugMode', false);
pickupNodes  = find(strcmp(tmpSys.nodeTypes, 'W_pickup'));
dropoffNodes = find(strcmp(tmpSys.nodeTypes, 'W_dropoff'));
clear tmpSys;

allTasks = cell(numRuns, 1);
for run = 1:numRuns
    rng(7345 + run);
    tasks = struct('sequentialId',  num2cell(1:numTasksToGenerate), ...
                   'pickup',        [], ...
                   'dropoff',       [], ...
                   'taskTypeId',    [], ...
                   'creationTime',  num2cell(zeros(1,numTasksToGenerate)), ...
                   'assignmentTime',num2cell(-ones(1,numTasksToGenerate)), ...
                   'assignedAGV',   num2cell(zeros(1,numTasksToGenerate)), ...
                   'totalDistance', num2cell(zeros(1,numTasksToGenerate)));
    for t = 1:numTasksToGenerate
        pIdx = randi(length(pickupNodes));
        dIdx = randi(length(dropoffNodes));
        tasks(t).pickup     = pickupNodes(pIdx);
        tasks(t).dropoff    = dropoffNodes(dIdx);
        tasks(t).taskTypeId = (pIdx - 1) * length(dropoffNodes) + dIdx;
    end
    allTasks{run} = tasks;
end
fprintf('Task pre-generati.\n\n');

%% ---- RACCOLTA RISULTATI (cell arrays, una cella per run) ----
idrrResults = cell(numRuns, 1);
rlResults   = cell(numRuns, 1);

%% ---- LOOP PARALLELO ----
fprintf('Avvio parfor...\n');
totalStart = tic;

parfor run = 1:numRuns

    fixedTasks = allTasks{run};

    % --------------------------------------------------
    % Oggetti locali al worker
    % --------------------------------------------------
    sys = AGV_System_IDRR_RL(nAGV, maxTime, maxTasks, taskGenRate, ...
        'enableVisualization', false, 'operatingMode', 'IDRR', ...
        'verboseLogging', false, 'debugMode', false);
    env = AGV_Environment(sys, 'operatingMode', 'IDRR', ...
        'enableShuffling', false, 'verboseLogging', false, 'debugMode', false);
    sys.connectToEnvironment(env);

    agent = MAPPO_Agent(nAGV, ...
        'learningRate', 3e-4, 'epsilon', 0.2, 'gamma', 0.99, ...
        'lambda', 0.95, 'valueCoeff', 0.5, 'entropyCoeff', 0.02, ...
        'maxGradNorm', 0.5, 'verboseLogging', false, 'debugMode', false);
    agent.loadAgent(modelPath);
    env.mappoAgent = agent;

    % --------------------------------------------------
    % FIFO
    % --------------------------------------------------
    idrrRes = struct();
    try
        sys.resetSystem();
        sys.resetForNewEpisode();
        sys.setOperatingMode('IDRR');

        rng(42);
        sys.taskQueue      = fixedTasks;
        sys.taskCounter    = numTasksToGenerate;
        sys.tasksGenerated = 0;
        sys.maxTime        = maxTime;
        sys.maxTasks       = maxTasks;

        sys.initializeAGVs();
        sys.runSimulation();

        m = sys.metrics.finalMetrics;
        mc = sys.metrics.maneuverCounts;

        % avgResolvingTime da agvStateTimes (corretto dopo fix classe)
        totalRes = 0;
        for i = 1:nAGV
            totalRes = totalRes + sys.agvStateTimes(i).Resolving;
        end

        idrrRes.productivity        = m.productivity;
        idrrRes.avgDistance         = m.avgDistancePerAGV;
        idrrRes.avgWaitingTime      = m.avgWaitingTime;
        idrrRes.avgResolvingTime    = totalRes / nAGV;
        idrrRes.totalConflicts      = m.totalConflicts;
        idrrRes.totalManeuvers      = m.totalManeuvers;
        idrrRes.res1                = mc.res1;
        idrrRes.res2                = mc.res2;
        idrrRes.avgTaskExecutionTime= m.avgExecutionTime;
        idrrRes.taskDelay_mean      = m.taskDelay_mean;
        idrrRes.taskDelay_std       = m.taskDelay_std;
        idrrRes.tasksCompleted      = length(sys.completedTasks);
        idrrRes.tasksPerAGV         = m.tasksPerAGV_vector;
        idrrRes.distancePerAGV      = m.distancePerAGV_vector;
        idrrRes.ok                  = true;
    catch ME
        idrrRes.ok  = false;
        idrrRes.err = ME.message;
        if strcmp(ME.identifier, 'AGV_DEADLOCK:AllWaiting')
            fprintf('⚠️  FIFO  run %3d — DEADLOCK  %s\n', run, ME.message);
        else
            fprintf('❌  FIFO  run %3d — ERRORE: %s\n', run, ME.message);
        end
    end

    % --------------------------------------------------
    % RL (MAPPO)
    % --------------------------------------------------
    rlRes = struct();
    try
        agent.deterministicMode = true;

        sys.resetSystem();
        sys.resetForNewEpisode();
        sys.setOperatingMode('RL_TESTING', env);
        env.resetTraining();
        env.setOperatingMode('RL_TESTING');

        rng(42);
        sys.taskQueue      = fixedTasks;
        sys.taskCounter    = numTasksToGenerate;
        sys.tasksGenerated = 0;
        sys.maxTime        = maxTime;
        sys.maxTasks       = maxTasks;

        sys.initializeAGVs();
        sys.runSimulation();

        m = sys.metrics.finalMetrics;
        mc = sys.metrics.maneuverCounts;

        totalRes = 0;
        for i = 1:nAGV
            totalRes = totalRes + sys.agvStateTimes(i).Resolving;
        end

        rlRes.productivity        = m.productivity;
        rlRes.avgDistance         = m.avgDistancePerAGV;
        rlRes.avgWaitingTime      = m.avgWaitingTime;
        rlRes.avgResolvingTime    = totalRes / nAGV;
        rlRes.totalConflicts      = m.totalConflicts;
        rlRes.totalManeuvers      = m.totalManeuvers;
        rlRes.res1                = mc.res1;
        rlRes.res2                = mc.res2;
        rlRes.avgTaskExecutionTime= m.avgExecutionTime;
        rlRes.taskDelay_mean      = m.taskDelay_mean;
        rlRes.taskDelay_std       = m.taskDelay_std;
        rlRes.tasksCompleted      = length(sys.completedTasks);
        rlRes.tasksPerAGV         = m.tasksPerAGV_vector;
        rlRes.distancePerAGV      = m.distancePerAGV_vector;
        rlRes.ok                  = true;
    catch ME
        rlRes.ok  = false;
        rlRes.err = ME.message;
        if strcmp(ME.identifier, 'AGV_DEADLOCK:AllWaiting')
            fprintf('⚠️  RL    run %3d — DEADLOCK  %s\n', run, ME.message);
        else
            fprintf('❌  RL    run %3d — ERRORE: %s\n', run, ME.message);
        end
    end

    idrrResults{run} = idrrRes;
    rlResults{run}   = rlRes;
end

fprintf('Parfor completato in %.1f min.\n\n', toc(totalStart)/60);

%% ---- AGGREGAZIONE RISULTATI ----
fprintf('Aggregazione...\n\n');

fields = {'productivity','avgDistance','avgWaitingTime','avgResolvingTime', ...
          'totalConflicts','totalManeuvers','res1','res2', ...
          'avgTaskExecutionTime','taskDelay_mean','taskDelay_std','tasksCompleted'};

idrrAll = struct(); rlAll = struct();
for f = fields
    idrrAll.(f{1}) = zeros(numRuns,1);
    rlAll.(f{1})   = zeros(numRuns,1);
end
idrrAll.tasksPerAGV    = zeros(numRuns, nAGV);
idrrAll.distancePerAGV = zeros(numRuns, nAGV);
rlAll.tasksPerAGV      = zeros(numRuns, nAGV);
rlAll.distancePerAGV   = zeros(numRuns, nAGV);

nFailIdrr = 0; nFailRl = 0;
for run = 1:numRuns
    if idrrResults{run}.ok
        for f = fields
            idrrAll.(f{1})(run) = idrrResults{run}.(f{1});
        end
        idrrAll.tasksPerAGV(run,:)    = idrrResults{run}.tasksPerAGV;
        idrrAll.distancePerAGV(run,:) = idrrResults{run}.distancePerAGV;
    else
        nFailIdrr = nFailIdrr + 1;
        fprintf('FIFO run %d FALLITA: %s\n', run, idrrResults{run}.err);
    end
    if rlResults{run}.ok
        for f = fields
            rlAll.(f{1})(run) = rlResults{run}.(f{1});
        end
        rlAll.tasksPerAGV(run,:)    = rlResults{run}.tasksPerAGV;
        rlAll.distancePerAGV(run,:) = rlResults{run}.distancePerAGV;
    else
        nFailRl = nFailRl + 1;
        fprintf('RL   run %d FALLITA: %s\n', run, rlResults{run}.err);
    end
end

%% ---- STAMPA TABELLA ----
fprintf('=================================================================\n');
fprintf('RISULTATI TABLE 2 — %d RUN (FIFO vs MAPPO)\n', numRuns);
fprintf('=================================================================\n');
fprintf('%-34s | %16s | %16s | %7s\n', 'METRICA', 'FIFO (mean±std)', 'MAPPO (mean±std)', 'Delta%');
fprintf('-----------------------------------------------------------------\n');

labels = {
    'Productivity [task/h]',        'productivity',         true;
    'Avg AGV Distance [m]',         'avgDistance',          false;
    'Avg Waiting Time [s]',         'avgWaitingTime',       false;
    'Avg Resolving Time [s]',       'avgResolvingTime',     false;
    'Total Conflicts',               'totalConflicts',       false;
    'Total Maneuvers',               'totalManeuvers',       false;
    '  of which Res1',               'res1',                false;
    '  of which Res2',               'res2',                false;
    'Avg Exec Time [s]',             'avgTaskExecutionTime', false;
    'Task Delay in Pool mean [s]',   'taskDelay_mean',       false;
    'Tasks Completed',               'tasksCompleted',       true;
};

for i = 1:size(labels,1)
    lbl   = labels{i,1};
    fld   = labels{i,2};
    higherIsBetter = labels{i,3};
    mu_f  = mean(idrrAll.(fld));  sd_f = std(idrrAll.(fld));
    mu_r  = mean(rlAll.(fld));    sd_r = std(rlAll.(fld));
    if mu_f ~= 0
        if higherIsBetter
            dlt = (mu_r - mu_f) / mu_f * 100;
        else
            dlt = (mu_f - mu_r) / mu_f * 100;
        end
    else
        dlt = 0;
    end
    sgn = '+'; if dlt < 0, sgn = ''; end
    fprintf('%-34s | %7.2f ± %6.2f | %7.2f ± %6.2f | %s%.2f%%\n', ...
        lbl, mu_f, sd_f, mu_r, sd_r, sgn, dlt);
end

fprintf('-----------------------------------------------------------------\n');
fprintf('Per-AGV Tasks (FIFO / MAPPO / Delta%%):\n');
for a = 1:nAGV
    mu_f = mean(idrrAll.tasksPerAGV(:,a));  sd_f = std(idrrAll.tasksPerAGV(:,a));
    mu_r = mean(rlAll.tasksPerAGV(:,a));    sd_r = std(rlAll.tasksPerAGV(:,a));
    dlt  = (mu_r - mu_f) / mu_f * 100;
    fprintf('  AGV%d: %5.2f±%4.2f / %5.2f±%4.2f / +%.2f%%\n', ...
        a, mu_f, sd_f, mu_r, sd_r, dlt);
end
fprintf('  CV FIFO=%.2f%%  CV MAPPO=%.2f%%\n', ...
    std(mean(idrrAll.tasksPerAGV,1))/mean(mean(idrrAll.tasksPerAGV,1))*100, ...
    std(mean(rlAll.tasksPerAGV,1))/mean(mean(rlAll.tasksPerAGV,1))*100);
fprintf('=================================================================\n');
if nFailIdrr>0||nFailRl>0
    fprintf('ATTENZIONE: %d run FIFO e %d run RL fallite.\n',nFailIdrr,nFailRl);
end

%% ---- SALVATAGGIO ----
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
save(sprintf('TABLE_5_&_6_7AGV_Par_%dRuns_%s.mat', numRuns, timestamp), ...
    'idrrAll', 'rlAll', 'numRuns', 'maxTime', 'nAGV');
fprintf('Salvato: TABLE_5_&_6_7AGV_Par_%dRuns_%s.mat\n', numRuns, timestamp);