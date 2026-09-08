%% generate_task_lists.m
% Genera e salva le liste di 200 task (pickup, dropoff) per ogni episodio.
% Il parametro seedBase consente di generare set di task completamente diversi
% senza sovrascrivere quello di default (seedBase=0 → seed 1..N, seedBase=100 → 101..N+100).
%
% OUTPUT: task_lists_seed<seedBase>.mat

clear; clc;

%% ========== PARAMETRI ==========
numEpisodes  = 100;
maxTasks     = 200;
seedBase     = 49;       % <-- MODIFICA QUI per set diversi di task
                        % seedBase=0:   seed 1..20   (default originale)
                        % seedBase=100: seed 101..120
                        % seedBase=200: seed 201..220

outputFile   = sprintf('task_lists_seed%d.mat', seedBase);

%% ========== TOPOLOGIA NODI ==========
dummySys     = AGV_System_IDRR_RL(7, 3600, maxTasks, 0.05);
pickupNodes  = find(strcmp(dummySys.nodeTypes, 'W_pickup'));
dropoffNodes = find(strcmp(dummySys.nodeTypes, 'W_dropoff'));
clear dummySys;

fprintf('Pickup  nodes: %d\n', length(pickupNodes));
fprintf('Dropoff nodes: %d\n', length(dropoffNodes));

%% ========== GENERAZIONE LISTE ==========
episodeSeeds = seedBase + (1:numEpisodes)';
taskLists    = cell(numEpisodes, 1);

for ep = 1:numEpisodes
    rng(episodeSeeds(ep));
    pickupSeq  = pickupNodes( randi(length(pickupNodes),  maxTasks, 1));
    dropoffSeq = dropoffNodes(randi(length(dropoffNodes), maxTasks, 1));
    taskLists{ep} = [pickupSeq, dropoffSeq];   % [maxTasks x 2]
end

%% ========== SALVATAGGIO ==========
save(outputFile, 'taskLists','episodeSeeds','maxTasks', ...
     'pickupNodes','dropoffNodes','numEpisodes','seedBase');
fprintf('\nListe salvate in "%s"  (seedBase=%d, episodi=%d, task/ep=%d)\n', ...
        outputFile, seedBase, numEpisodes, maxTasks);