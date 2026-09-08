classdef AGV_System_IDRR_RL < handle
    
    properties
        % === PARAMETRI SISTEMA ===
        nAGV                    % Numero AGV
        currentTime             % Tempo corrente simulazione
        agvSpeed = 1.0          % Velocità AGV [m/s]
        timeStep = 0.1         % Passo temporale [s]
        maxTime                 % Tempo massimo simulazione
        
        % === LAYER SIMBOLICO ===
        nodes                   % Array di struct per nodi simbolici
        edges                   % Matrice di adiacenza con pesi (distanze)
        graph                   % Oggetto graph di MATLAB
        nodeNameMap             % Map nome->indice nodo
        indexNameMap            % Map indice->nome nodo
        nodePositions           % Posizioni [x,y] dei nodi simbolici
        nodeTypes               % Tipi nodi ('P', 'W_pickup', 'W_dropoff', 'T', 'X')
        
        % === LAYER GEOMETRICO ===
        controlPoints           % Map nodeIdx -> struct CP data
        cpPositions             % Matrice [N_cp x 2] posizioni tutti i CP
        cpNodeMap               % Map cpIdx -> nodeIdx di appartenenza
        geometricGraph          % Grafo delle connessioni tra CP
        geometricEdges          % Matrice adiacenza CP con pesi
        
        % === ROUTING ===
        cpRoutingTable          % Map 'fromCP_toCP' -> path di CP
        symbolicRoutingTable    % Routing simbolico
        
        % === STRUTTURA AGV ===
        AGVs                    % Array di struct AGV
        
        % === STATI SISTEMA ===
        nodeStates              % Array [nNodes x 1]: numero AGV in ogni nodo
        nodeOccupants           % Cell array: AGV presenti in ogni nodo
        cpStates                % Array [nCPs x 1]: 0=libero, 1=occupato
        cpOccupants             % Array [nCPs x 1]: ID AGV che occupa il CP
        
        % === GESTIONE TASK (Algorithm 1) ===
        taskQueue               % Lista task pendenti (corrisponde ad A in Algorithm 1)
        completedTasks          % Task completati
        taskCounter = 0         % Contatore task generati
        taskGenerationRate      % Task per secondo
        maxTasks                % Numero massimo task da generare
        lastTaskTime = 0        % Ultimo tempo di generazione task
        tasksGenerated = 0      % Numero task già generati
        preloadedTaskList = []  % Lista task pre-caricati [pickup, dropoff] per testing riproducibile (IDRR baseline)
        preloadedTaskIdx  = 0   % Indice corrente nella lista pre-caricata
        
        % === CONFLICT TRACKING ===
        activeConflicts         % Map per evitare conteggi multipli conflitti

        % === PRECALCOLO CICLI ===
        allCycles              % Cell array di tutti i cicli nel grafo
        nodeCycleMap          % Map nodeIdx -> array di indici cicli che contengono il nodo
        
        % === SHARED ROUTES (Algorithm 2) ===
        sharedRoutes            % Map 'agvId1_agvId2' -> shared route
        
        % === MANOVRE E CONFLITTI ===
        pendingManeuvers        % Array di struct per manovre pendenti
        res1Duration = 4.0          % Durata manovra Res1 [s]
        res2Duration = 8.0          % Durata manovra Res2 [s]
        activeManeuvers             % Array di struct per manovre in corso

        % === METRICHE ===
        metrics                 % Struct per raccolta metriche
        
        % === DEBUG E VISUALIZZAZIONE ===
        debugMode = false
        figHandles              % Handle per figure di visualizzazione
    
        % === VISUALIZZAZIONE E DEBUG ===
        visualizer              % Istanza IDRRVisualizer per visualizzazione real-time
        visualizationEnabled = false  % Flag per abilitare visualizzazione
    
        % ============= ESTENSIONE RL =============

        % === MODALITÀ OPERATIVE ===
        operatingMode = 'IDRR'      % 'IDRR' | 'RL_TRAINING' | 'RL_TESTING'

        % === TOP-5 TASK QUEUE ===
        top5TaskQueue = []          % Primi 5 task dal pool globale

        % === THROUGHPUT TRACKING ===
        lastCompletedTaskTime = 0   % Timestamp ultimo task completato
        lastThroughputValue = 0     % Throughput quando ultimo task completato
        currentThroughputValue = 0  % Throughput attuale
        throughputDerivative = 0    % Derivata semplice throughput

        % === METRICHE EPISODICHE ===
        episodeStartTime = 0        % Timestamp inizio episodio
        episodeTasksCompleted = 0   % Task completati nell'episodio

        % === STATE TRACKING DETTAGLIATO ===
        agvStateHistory = []        % Storico dettagliato cambi stati AGV
        agvStateTimes = []          % Tempi cumulativi in ogni stato per AGV

        % === NUOVE METRICHE PER TESTING ===
        taskPoolEntryTimes      % Map: taskId -> timestamp ingresso pool
        taskDelaysInPool        % Array dei delay di tutti i task completati [s]

        % === RL EVENT MANAGEMENT ===
        pendingRLDecisions = []     % Array di AGV IDs che aspettano decisione RL
        rlDecisionTimestamps = []   % Timestamp delle decisioni RL

        % === INTERFACCIA RL ===
        environmentInterface = []   % Handle a AGV_Environment

        % === LOGGING ===
        verboseLogging = false     % Flag per logging dettagliato RL

        % === PRIM TASK ALLOCATION (Algorithm 3 - Cap. 5) ===
        primAuctionPool         % Array struct: task nel pool di asta (TN ∪ TU)
        preGeneratedTasks       % Array struct: lista completa task pre-generati
        preGenTaskIndex = 1     % Indice: prossimo task da iniettare dalla lista
        taskInjectionInterval = 20.0  % Intervallo iniezione task [s]
        lastInjectionTime = 0   % Timestamp ultima iniezione

        % === DEADLOCK DETECTION ===
        enableDeadlockDetection = true  % Flag per abilitare controllo deadlock
        deadlockCounter         = 0    % Contatore persistenza: # check consecutivi con tutti in WAITING
        
        % === PICKUP STATION PRODUCTIVITY ===
        pickupStationCounts     % Map: pickupNode -> numero task completati

        % === EVENT-DRIVEN TASK GENERATION ===
        eventDrivenMode = false         % Attiva generazione task su evento pickup
        eventDropoffSequences = []      % [nPickups x maxEventi]: dropoff pre-generati
        eventPickupNodes = []           % Indici nodi pickup (corrispondono alle righe)
        eventPickupCounters             % Map: pickupNode -> n. task già generati
    
    end
    
    methods
        function obj = AGV_System_IDRR_RL(nAGV, maxTime, maxTasks, taskGenRate, varargin)
            
            % Crea e configura parser
            p = inputParser;

            % Parametri obbligatori
            addRequired(p, 'nAGV', @isnumeric);
            addRequired(p, 'maxTime', @isnumeric);
            addRequired(p, 'maxTasks', @isnumeric);
            addRequired(p, 'taskGenRate', @isnumeric);

            % Parametri opzionali
            addParameter(p, 'enableVisualization', false, @islogical);
            addParameter(p, 'enableVideoRecording', false, @islogical);
            addParameter(p, 'operatingMode', 'IDRR', @ischar);
            addParameter(p, 'verboseLogging', false, @islogical);
            addParameter(p, 'debugMode', false, @islogical);

            % Parse degli input
            parse(p, nAGV, maxTime, maxTasks, taskGenRate, varargin{:});

            % Estrai risultati
            enableVisualization = p.Results.enableVisualization;
            enableVideoRecording = p.Results.enableVideoRecording;
            obj.operatingMode = p.Results.operatingMode;
            obj.verboseLogging = p.Results.verboseLogging;
            obj.debugMode = p.Results.debugMode;

            obj.nAGV = nAGV;
            obj.maxTime = maxTime;
            obj.maxTasks = maxTasks;
            obj.taskGenerationRate = taskGenRate;
            obj.currentTime = 0;
            obj.figHandles = struct();

            % Inizializza tutto il sistema
            obj.initializeSymbolicLayer();
            obj.initializeGeometricLayer();
            obj.computeGeometricConnections();
            obj.computeGeometricRouting();
            obj.initializeSystemStates();
            obj.initializeAGVs();
            obj.initializeTaskManagement();
            obj.initializeSharedRoutes();
            obj.initializeMetrics();
            obj.initializeRLComponents();
            
            if enableVisualization || enableVideoRecording
                obj.initializeVisualization(enableVideoRecording);
            end

            %fprintf('Sistema IDRR-RL inizializzato: %d AGV, %d nodi simbolici, %d CP - Modalità: %s\n', ...
            %nAGV, length(obj.nodes), size(obj.cpPositions, 1), obj.operatingMode);

            if obj.visualizationEnabled
                if enableVideoRecording
                    %fprintf('🎥 Visualizzazione + Video Recording: ABILITATI\n');
                else
                    %fprintf('🎬 Visualizzazione real-time: ABILITATA\n');
                end
            end
        end
     
        % === INIZIALIZZAZIONE LAYER SIMBOLICO ===
        function initializeSymbolicLayer(obj)
            % Mantiene la logica esistente per il layer simbolico
            obj.initializeLayoutFromPaper();
            obj.createNodeNameMapping();
            obj.buildSymbolicAdjacency();
            obj.initializeSymbolicNodes();
            obj.computeSymbolicRouting();
            obj.verifyRouteSymmetry();
            obj.precomputeAllCycles();
        end
        
        function initializeLayoutFromPaper(obj)
            % Layout esatto dal paper - Figura 1
            obj.nodePositions = [
                % Parcheggi P1-P10
                100, 141;   130, 141;   45, 90;     154, 126;   103, 127;   
                103, 120;   103, 110;   103, 100;   103, 90;    103, 80;    
                
                % Stazioni Pickup W1-W7
                0, 121;     93, 121;    65, 95;     40, 10;     154, 85;    
                123, 45;    154, 10;    
                
                % Stazioni Dropoff W8-W9
                154, 110;   0, 10;      
                
                % Intersezioni T1-T29
                10, 121;    10, 70;     35, 131;    35, 70;     35, 90;     
                75, 70;     50, 70;     10, 10;     50, 10;     50, 0;      
                113, 0;     113, 45;    144, 10;    144, 70;    144, 85;    
                144, 110;   144, 126;   130, 131;   113, 131;   100, 131;   
                93, 131;    75, 131;    113, 127;   113, 120;   113, 110;   
                113, 100;   113, 90;    113, 80;    75, 95;     
                
                % Intersezione Centrale X1
                113, 70     
            ];
            
            obj.nodeTypes = [
                repmat({'P'}, 10, 1);              
                repmat({'W_pickup'}, 7, 1);        
                repmat({'W_dropoff'}, 2, 1);       
                repmat({'T'}, 29, 1);              
                {'X'}                              
            ];
        end
        
        function createNodeNameMapping(obj)
            obj.nodeNameMap = containers.Map();
            obj.indexNameMap = containers.Map('KeyType', 'int32', 'ValueType', 'char');
            
            % Parcheggi P1-P10
            for i = 1:10
                name = sprintf('P%d', i);
                obj.nodeNameMap(name) = i;
                obj.indexNameMap(i) = name;
            end
            
            % Stazioni W1-W9
            for i = 1:9
                name = sprintf('W%d', i);
                idx = i + 10;
                obj.nodeNameMap(name) = idx;
                obj.indexNameMap(idx) = name;
            end
            
            % Intersezioni T1-T29
            for i = 1:29
                name = sprintf('T%d', i);
                idx = i + 19;
                obj.nodeNameMap(name) = idx;
                obj.indexNameMap(idx) = name;
            end
            
            % Intersezione X1
            obj.nodeNameMap('X1') = 49;
            obj.indexNameMap(49) = 'X1';
        end
        
        function buildSymbolicAdjacency(obj)
            nNodes = size(obj.nodePositions, 1);
            obj.edges = zeros(nNodes, nNodes);
            
            % Connessioni esatte dal paper
            connections = [
                % Stazioni agli incroci
                1, 39;  2, 37;  3, 24;  4, 36;  5, 42;  6, 43;  7, 44;  8, 45;  9, 46;  10, 47;
                11, 20; 12, 40; 13, 48; 14, 28; 15, 34; 16, 31; 17, 32; 18, 35; 19, 27;
                
                % Connessioni principali
                20, 22; 22, 41; 41, 40; 40, 39; 39, 38; 38, 37; 37, 36;
                21, 23; 23, 26; 26, 25; 25, 49; 49, 33;
                27, 29; 29, 30; 30, 32;
                20, 21; 21, 27; 22, 24; 24, 23; 26, 28; 28, 29;
                41, 48; 48, 25;
                38, 42; 42, 43; 43, 44; 44, 45; 45, 46; 46, 47; 47, 49; 49, 31; 31, 30;
                36, 35; 35, 34; 34, 33; 33, 32;
            ];
            
            % Calcola distanze euclidee
            for i = 1:size(connections, 1)
                n1 = connections(i, 1);
                n2 = connections(i, 2);
                if n1 <= nNodes && n2 <= nNodes
                    dist = norm(obj.nodePositions(n1,:) - obj.nodePositions(n2,:));
                    obj.edges(n1, n2) = dist;
                    obj.edges(n2, n1) = dist;
                end
            end
            
            obj.graph = graph(obj.edges);
        end
        
        function initializeSymbolicNodes(obj)
            nNodes = size(obj.nodePositions, 1);
            obj.nodes = repmat(struct(), nNodes, 1);
            
            for i = 1:nNodes
                nodeType = obj.nodeTypes{i};
                
                obj.nodes(i).id = i;
                obj.nodes(i).type = nodeType;
                obj.nodes(i).position = obj.nodePositions(i,:);
                obj.nodes(i).name = obj.getNodeName(i);
                
                % Capacità secondo paper
                switch nodeType
                    case 'P'
                        obj.nodes(i).capacity = 1;
                    case {'W_pickup', 'W_dropoff'}
                        obj.nodes(i).capacity = 1;
                    case 'T'
                        obj.nodes(i).capacity = 3;
                    case 'X'
                        obj.nodes(i).capacity = 4;
                end
                
                obj.nodes(i).occupancy = 0;
                obj.nodes(i).agvsPresent = [];
            end
        end
        
        function computeSymbolicRouting(obj)
            % Versione ottimizzata che sfrutta la simmetria dei percorsi

            obj.symbolicRoutingTable = containers.Map();

            parkingNodes = find(strcmp(obj.nodeTypes, 'P'));
            pickupNodes = find(strcmp(obj.nodeTypes, 'W_pickup'));
            dropoffNodes = find(strcmp(obj.nodeTypes, 'W_dropoff'));

            % Set per tenere traccia delle coppie già calcolate
            calculatedPairs = containers.Map();

            %fprintf('Calcolo routing simbolico ottimizzato...\n');

            % === MACRO 1: P → W_pickup ===
            for fromPark = parkingNodes'
                for toPickup = pickupNodes'
                    [path, ~] = obj.getOrCalculatePath(fromPark, toPickup, calculatedPairs);
                    key = sprintf('%d_%d', fromPark, toPickup);
                    obj.symbolicRoutingTable(key) = path;
                end
            end

            % === MACRO 2: W_pickup → W_dropoff ===
            for fromPickup = pickupNodes'
                for toDropoff = dropoffNodes'
                    [path, ~] = obj.getOrCalculatePath(fromPickup, toDropoff, calculatedPairs);
                    key = sprintf('%d_%d', fromPickup, toDropoff);
                    obj.symbolicRoutingTable(key) = path;
                end
            end

            % === MACRO 3: W_dropoff → W_pickup ===
            for fromDropoff = dropoffNodes'
                for toPickup = pickupNodes'
                    % Qui sfrutto la simmetria: se ho già W_pickup → W_dropoff,
                    % uso quello invertito invece di ricalcolare
                    [path, wasReversed] = obj.getOrCalculatePath(fromDropoff, toPickup, calculatedPairs);
                    key = sprintf('%d_%d', fromDropoff, toPickup);
                    obj.symbolicRoutingTable(key) = path;

                    if wasReversed && obj.debugMode
                        % fprintf('Riutilizzato percorso invertito: %s → %s\n', ...
                        %     obj.getNodeName(fromDropoff), obj.getNodeName(toPickup));
                    end
                end
            end

            % === MACRO 4: W_dropoff → P ===
            for fromDropoff = dropoffNodes'
                for toPark = parkingNodes'
                    [path, ~] = obj.getOrCalculatePath(fromDropoff, toPark, calculatedPairs);
                    key = sprintf('%d_%d', fromDropoff, toPark);
                    obj.symbolicRoutingTable(key) = path;
                end
            end
            
            if obj.debugMode
                fprintf('Routing simbolico: %d path necessari\n', length(keys(obj.symbolicRoutingTable)));
                fprintf('  P→W_pickup: %d\n', length(parkingNodes) * length(pickupNodes));
                fprintf('  W_pickup→W_dropoff: %d\n', length(pickupNodes) * length(dropoffNodes));
                fprintf('  W_dropoff→W_pickup: %d\n', length(dropoffNodes) * length(pickupNodes));
                fprintf('  W_dropoff→P: %d\n', length(dropoffNodes) * length(parkingNodes));
            end
        end

        function [path, wasReversed] = getOrCalculatePath(obj, fromNode, toNode, calculatedPairs)
            % Ottiene un percorso, calcolandolo o riutilizzando quello inverso

            wasReversed = false;

            % Crea chiavi per entrambe le direzioni
            key_direct = sprintf('%d_%d', fromNode, toNode);
            key_reverse = sprintf('%d_%d', toNode, fromNode);

            % Controlla se esiste già il percorso diretto
            if calculatedPairs.isKey(key_direct)
                path = calculatedPairs(key_direct);
                return;
            end

            % Controlla se esiste il percorso inverso
            if calculatedPairs.isKey(key_reverse)
                reversePath = calculatedPairs(key_reverse);
                path = fliplr(reversePath);  % Inverti il percorso
                wasReversed = true;
                return;
            end

            % Nessun percorso esistente, calcola nuovo
            path = shortestpath(obj.graph, fromNode, toNode);
            calculatedPairs(key_direct) = path;
        end

        function verifyRouteSymmetry(obj)
            % Verifica che i percorsi inversi siano effettivamente simmetrici

            %fprintf('\n=== VERIFICA SIMMETRIA PERCORSI ===\n');

            % Test su alcune coppie di nodi
            pickupNodes = find(strcmp(obj.nodeTypes, 'W_pickup'));
            dropoffNodes = find(strcmp(obj.nodeTypes, 'W_dropoff'));

            asymmetricCount = 0;
            totalTests = 0;

            % Test W_pickup ↔ W_dropoff
            for i = 1:min(3, length(pickupNodes))  % Testa solo le prime 3 per brevità
                for j = 1:min(2, length(dropoffNodes))
                    fromPickup = pickupNodes(i);
                    toDropoff = dropoffNodes(j);

                    % Calcola percorsi
                    key1 = sprintf('%d_%d', fromPickup, toDropoff);
                    key2 = sprintf('%d_%d', toDropoff, fromPickup);

                    if obj.symbolicRoutingTable.isKey(key1) && obj.symbolicRoutingTable.isKey(key2)
                        path1 = obj.symbolicRoutingTable(key1);
                        path2 = obj.symbolicRoutingTable(key2);
                        path2_reversed = fliplr(path2);  % Inverti path2

                        totalTests = totalTests + 1;

                        if ~isequal(path1, path2_reversed) && obj.debugMode
                            asymmetricCount = asymmetricCount + 1;
                            %fprintf('ASIMMETRIA TROVATA:\n');
                            %fprintf('  %s → %s: [%s]\n', obj.getNodeName(fromPickup), ...
                            %    obj.getNodeName(toDropoff), ...
                            %    strjoin(arrayfun(@(x) obj.getNodeName(x), path1, 'UniformOutput', false), ', '));
                            %fprintf('  %s → %s: [%s]\n', obj.getNodeName(toDropoff), ...
                            %    obj.getNodeName(fromPickup), ...
                            %    strjoin(arrayfun(@(x) obj.getNodeName(x), path2, 'UniformOutput', false), ', '));
                            %fprintf('  Invertito:     [%s]\n', ...
                            %    strjoin(arrayfun(@(x) obj.getNodeName(x), path2_reversed, 'UniformOutput', false), ', '));
                        else
                            % fprintf('✓ Simmetrico: %s ↔ %s\n', ...
                            %     obj.getNodeName(fromPickup), obj.getNodeName(toDropoff));
                        end
                    end
                end
            end

            % Test aggiuntivo: verifica che la matrice edges sia simmetrica
            if ~issymmetric(obj.edges)
                %fprintf('⚠️  ATTENZIONE: La matrice edges NON è simmetrica!\n');
            else
                %fprintf('✓ Matrice edges è simmetrica\n');
            end

            % fprintf('\nRisultati: %d/%d percorsi simmetrici (%.1f%%)\n', ...
            %     totalTests - asymmetricCount, totalTests, ...
            %     100 * (totalTests - asymmetricCount) / max(totalTests, 1));

            if asymmetricCount == 0
                %fprintf('🎯 Tutti i percorsi sono simmetrici come atteso!\n');
            else
                %fprintf('❌ Trovate %d asimmetrie - possibile problema nell''implementazione\n', asymmetricCount);
            end
        end
        
        % === INIZIALIZZAZIONE LAYER GEOMETRICO ===
        function initializeGeometricLayer(obj)
            if obj.debugMode
                fprintf('Inizializzazione layer geometrico...\n');
                fprintf('Nodi disponibili: %d\n', length(obj.nodes));
            end
            
            obj.controlPoints = containers.Map('KeyType', 'int32', 'ValueType', 'any');
            obj.cpNodeMap = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
            
            allCPPositions = [];
            cpCounter = 0;
            
            nNodes = length(obj.nodes);
            if nNodes == 0
                error('Nessun nodo simbolico trovato! Problema nell''inizializzazione simbolica.');
            end
            
            for nodeIdx = 1:nNodes
                node = obj.nodes(nodeIdx);
                
                if obj.debugMode && nodeIdx <= 5  % Mostra solo i primi 5 per non riempire il log
                    fprintf('Generando CP per nodo %d (%s)...\n', nodeIdx, node.type);
                end
                
                % Genera posizioni CP per questo nodo
                cpPositions = obj.generateCPPositions(node);
                [nCPs, ~] = size(cpPositions);
                
                if nCPs == 0
                    error('Nessun CP generato per nodo %d (%s)', nodeIdx, node.type);
                end
                
                % Salva CP data per questo nodo
                cpData = struct();
                cpData.nodeId = nodeIdx;
                cpData.nodeType = node.type;
                cpData.positions = cpPositions;
                cpData.nCPs = nCPs;
                cpData.globalIndices = (cpCounter + 1):(cpCounter + nCPs);
                cpData.states = zeros(1, nCPs);  % 0=libero, 1=occupato/riservato
                cpData.occupiedBy = zeros(1, nCPs);  % ID AGV che occupa
                
                obj.controlPoints(nodeIdx) = cpData;
                
                % Aggiorna strutture globali
                allCPPositions = [allCPPositions; cpPositions];
                for i = 1:nCPs
                    cpIdx = cpCounter + i;
                    obj.cpNodeMap(cpIdx) = nodeIdx;
                end
                
                cpCounter = cpCounter + nCPs;
            end
            
            obj.cpPositions = allCPPositions;
            
            if obj.debugMode
                fprintf('Layer geometrico: %d CP totali generati\n', length(obj.cpPositions));
                if length(obj.cpPositions) == 0
                    error('Nessun CP generato in totale!');
                end
                obj.printCPSummary();
            end
        end
        
        function positions = generateCPPositions(obj, node)
            % Usa coordinate predefinite ottimizzate
            nodeIdx = node.id;
            positions = obj.getPredefinedCPCoordinates(nodeIdx);
        end

        function coordinates = getPredefinedCPCoordinates(obj, nodeIdx)
            % Coordinate fisse ottimizzate per ogni nodo

            switch nodeIdx
                % === PARCHEGGI P1-P10 ===
                case 1,  coordinates = [100, 141];    % P1
                case 2,  coordinates = [130, 141];    % P2
                case 3,  coordinates = [45, 90];      % P3
                case 4,  coordinates = [154, 126];    % P4
                case 5,  coordinates = [103, 127];    % P5
                case 6,  coordinates = [103, 120];    % P6
                case 7,  coordinates = [103, 110];    % P7
                case 8,  coordinates = [103, 100];    % P8
                case 9,  coordinates = [103, 90];     % P9
                case 10, coordinates = [103, 80];     % P10

                    % === STAZIONI W1-W9 ===
                case 11, coordinates = [0, 121];      % W1
                case 12, coordinates = [93, 121];     % W2
                case 13, coordinates = [65, 95];      % W3
                case 14, coordinates = [40, 10];      % W4
                case 15, coordinates = [154, 85];     % W5
                case 16, coordinates = [123, 45];     % W6
                case 17, coordinates = [154, 10];     % W7
                case 18, coordinates = [154, 110];    % W8
                case 19, coordinates = [0, 10];       % W9

                    % === NODI T (3 CP ciascuno) ===
                case 20, coordinates = [7, 121; 13, 124; 10, 118];        % T1: W1,T3,T2
                case 21, coordinates = [13, 70; 10, 73; 10, 67];          % T2: T4,T1,T8
                case 22, coordinates = [35, 128; 38, 131; 32, 129];       % T3: T5,T22,T1
                case 23, coordinates = [38, 70; 32, 70; 35, 73];          % T4: T7,T2,T5
                case 24, coordinates = [35, 93; 38, 90; 32, 90];          % T5: T3,T4,P3
                case 25, coordinates = [72, 70; 78, 70; 75, 73];          % T6: T7,X1,T29
                case 26, coordinates = [47, 70; 53, 70; 50, 67];          % T7: T4,T6,T9
                case 27, coordinates = [10, 13; 13, 10; 7, 10];           % T8: T2,T10,W9
                case 28, coordinates = [50, 13; 53, 10; 47, 10];          % T9: T7,T10,W4
                case 29, coordinates = [47, 0; 53, 0; 50, 3];             % T10: T9,T11,T8
                case 30, coordinates = [110, 0; 113, 3; 116, 0];          % T11: T10,T12,T13
                case 31, coordinates = [110, 45; 113, 42; 113, 48];       % T12: W6,T11,X1
                case 32, coordinates = [141, 10; 144, 13; 147, 10];       % T13: T11,T14,W7
                case 33, coordinates = [144, 67; 144, 73; 141, 70];       % T14: T13,T15,X1
                case 34, coordinates = [144, 82; 144, 88; 147, 85];       % T15: T14,T16,W5
                case 35, coordinates = [144, 107; 144, 113; 147, 110];    % T16: T15,T17,W8
                case 36, coordinates = [144, 123; 144, 129; 147, 126];    % T17: T16,T18,P4
                case 37, coordinates = [127, 131; 133, 131; 130, 128];    % T18: T19,P2,T17
                case 38, coordinates = [110, 131; 113, 134; 116, 131];    % T19: T20,T23,T18
                case 39, coordinates = [97, 131; 100, 134; 103, 131];     % T20: T21,P1,T19
                case 40, coordinates = [96, 131; 93, 128; 90, 131];       % T21: T20,T22,W2
                case 41, coordinates = [72, 131; 78, 131; 75, 128];       % T22: T3,T21,T29
                case 42, coordinates = [110, 127; 113, 130; 113, 124];    % T23: P5,T19,T24
                case 43, coordinates = [110, 120; 113, 123; 113, 117];    % T24: P6,T23,T25
                case 44, coordinates = [110, 110; 113, 113; 113, 107];    % T25: P7,T24,T26
                case 45, coordinates = [110, 100; 113, 103; 113, 97];     % T26: P8,T25,T27
                case 46, coordinates = [110, 90; 113, 93; 113, 87];       % T27: P9,T26,T28
                case 47, coordinates = [110, 80; 113, 83; 113, 77];       % T28: P10,T27,X1
                case 48, coordinates = [72, 95; 75, 92; 75, 98];          % T29: W3,T6,T22

                    % === NODO X1 (4 CP) ===
                case 49, coordinates = [110, 70; 113, 67; 116, 70; 113, 73]; % X1: T6,T12,T14,T28

                otherwise
                    error('Coordinate CP non definite per nodo %d', nodeIdx);
            end
        end

        % === CONNESSIONI GEOMETRICHE ===
        function computeGeometricConnections(obj)
            nCPs = size(obj.cpPositions, 1);
            obj.geometricEdges = zeros(nCPs, nCPs);
            
            % 1. Connessioni INTRA-nodo
            obj.addIntraNodeConnections();
            
            % 2. Connessioni INTER-nodo
            obj.addInterNodeConnections();
            
            % Crea grafo geometrico
            obj.geometricGraph = graph(obj.geometricEdges);
            
            if obj.debugMode
                nConnections = nnz(obj.geometricEdges) / 2;  % Diviso 2 perché simmetrico
                fprintf('Connessioni geometriche: %d totali\n', nConnections);
            end
        end
        
        function addIntraNodeConnections(obj)
            nodeIndices = keys(obj.controlPoints);
            
            for i = 1:length(nodeIndices)
                nodeIdx = nodeIndices{i};
                cpData = obj.controlPoints(nodeIdx);
                nodeType = cpData.nodeType;
                globalIndices = cpData.globalIndices;
                
                switch nodeType
                    case {'P', 'W_pickup', 'W_dropoff'}
                        % Nodi WP: 1 solo CP, nessuna connessione interna
                        continue;
                        
                    case 'T'
                        % Nodi T: connessioni tra CP adiacenti (triangolo)
                        connections = [1, 2; 2, 3; 3, 1];
                        
                    case 'X'
                        % Nodi X: connessioni a quadrato + all-to-all
                        % Connessioni quadrato
                        connections = [1, 2; 2, 3; 3, 4; 4, 1];
                        
                        % Connessioni diagonali (all-to-all)
                        connections = [connections; 1, 3; 2, 4];
                end
                
                % Aggiungi connessioni alla matrice globale
                for j = 1:size(connections, 1)
                    cp1 = globalIndices(connections(j, 1));
                    cp2 = globalIndices(connections(j, 2));
                    
                    dist = norm(obj.cpPositions(cp1, :) - obj.cpPositions(cp2, :));
                    obj.geometricEdges(cp1, cp2) = dist;
                    obj.geometricEdges(cp2, cp1) = dist;
                end
            end
        end
        
        function addInterNodeConnections(obj)
            % Connessioni ottimizzate 1:1 senza intersezioni

            cpConnections = [
                % Formato: [node1, cp1_local, node2, cp2_local]

                % === WP → T ===
                1, 1, 39, 2;    % P1 → T20(Nord)
                2, 1, 37, 2;    % P2 → T18(Est)
                3, 1, 24, 3;    % P3 → T5(Ovest)
                4, 1, 36, 3;    % P4 → T17(Est)
                5, 1, 42, 1;    % P5 → T23(Ovest)
                6, 1, 43, 1;    % P6 → T24(Ovest)
                7, 1, 44, 1;    % P7 → T25(Ovest)
                8, 1, 45, 1;    % P8 → T26(Ovest)
                9, 1, 46, 1;    % P9 → T27(Ovest)
                10, 1, 47, 1;   % P10 → T28(Ovest)

                11, 1, 20, 1;   % W1 → T1(Ovest)
                12, 1, 40, 2;   % W2 → T21(Nord)
                13, 1, 48, 1;   % W3 → T29(Ovest)
                14, 1, 28, 3;   % W4 → T9(Ovest)
                15, 1, 34, 3;   % W5 → T15(Est)
                16, 1, 31, 1;   % W6 → T12(Ovest)
                17, 1, 32, 3;   % W7 → T13(Est)
                18, 1, 35, 3;   % W8 → T16(Est)
                19, 1, 27, 3;   % W9 → T8(Ovest)

                % === T-T e T-X ===
                20, 2, 22, 3;   % T1(NordEst) ↔ T3(SudOvest)
                20, 3, 21, 2;   % T1(Sud) ↔ T2(Nord)
                21, 1, 23, 2;   % T2(Est) ↔ T4(Ovest)
                21, 3, 27, 1;   % T2(Sud) ↔ T8(Nord)
                22, 1, 24, 1;   % T3(Sud) ↔ T5(Nord)
                22, 2, 41, 2;   % T3(Est) ↔ T22(Est)
                23, 1, 26, 2;   % T4(Est) ↔ T7(Est)
                23, 3, 24, 3;   % T4(Sud) ↔ T5(Nord)
                25, 1, 26, 2;   % T6(Ovest) ↔ T7(Est)
                25, 2, 49, 1;   % T6(Est) ↔ X1(Ovest)
                25, 3, 48, 3;   % T6(Nord) ↔ T29(Nord)
                26, 3, 28, 1;   % T7(Sud) ↔ T9(Nord)
                27, 2, 29, 1;   % T8(Est) ↔ T10(Ovest)
                28, 2, 29, 3;   % T9(Est) ↔ T10(Ovest)
                29, 2, 30, 1;   % T10(Est) ↔ T11(Ovest)
                30, 2, 31, 2;   % T11(Nord) ↔ T12(Sud)
                30, 3, 32, 1;   % T11(Nord) ↔ T13(Ovest)
                31, 3, 49, 2;   % T12(Nord) ↔ X1(Sud)
                32, 2, 33, 1;   % T13(Nord) ↔ T14(Sud)
                33, 2, 34, 1;   % T14(Nord) ↔ T15(Sud)
                33, 3, 49, 3;   % T14(Ovest) ↔ X1(Est)
                34, 2, 35, 1;   % T15(Nord) ↔ T16(Sud)
                35, 2, 36, 1;   % T16(Nord) ↔ T17(Sud)
                36, 2, 37, 3;   % T17(Nord) ↔ T18(Sud)
                37, 1, 38, 3;   % T18(Ovest) ↔ T19(Est)
                38, 1, 39, 3;   % T19(Ovest) ↔ T20(Est)
                38, 2, 42, 2;   % T19(Nord) ↔ T23(Nord)
                39, 1, 40, 1;   % T20(Ovest) ↔ T21(Est)
                40, 3, 41, 2;   % T21(Sud) ↔ T22(Est)
                41, 3, 48, 3;   % T22(Sud) ↔ T29(Nord)
                42, 3, 43, 2;   % T23(Sud) ↔ T24(Nord)
                43, 3, 44, 2;   % T24(Sud) ↔ T25(Nord)
                44, 3, 45, 2;   % T25(Sud) ↔ T26(Nord)
                45, 3, 46, 2;   % T26(Sud) ↔ T27(Nord)
                46, 3, 47, 2;   % T27(Sud) ↔ T28(Nord)
                47, 3, 49, 4;   % T28(Sud) ↔ X1(Nord)
                ];

            % Applica le connessioni (codice esistente invariato)
            for i = 1:size(cpConnections, 1)
                node1 = cpConnections(i, 1);
                cp1_local = cpConnections(i, 2);
                node2 = cpConnections(i, 3);
                cp2_local = cpConnections(i, 4);

                if node1 <= length(obj.nodes) && node2 <= length(obj.nodes)
                    cpData1 = obj.controlPoints(node1);
                    cpData2 = obj.controlPoints(node2);

                    if cp1_local <= cpData1.nCPs && cp2_local <= cpData2.nCPs
                        cp1_global = cpData1.globalIndices(cp1_local);
                        cp2_global = cpData2.globalIndices(cp2_local);

                        dist = norm(obj.cpPositions(cp1_global, :) - obj.cpPositions(cp2_global, :));
                        obj.geometricEdges(cp1_global, cp2_global) = dist;
                        obj.geometricEdges(cp2_global, cp1_global) = dist;
                    end
                end
            end
        end
       
        % === ROUTING GEOMETRICO ===
        function computeGeometricRouting(obj)
            obj.cpRoutingTable = containers.Map();
            
            % Per ogni path simbolico, calcola entry/exit CP per ogni nodo
            symbolicKeys = keys(obj.symbolicRoutingTable);
            
            for i = 1:length(symbolicKeys)
                key = symbolicKeys{i};
                symbolicPath = obj.symbolicRoutingTable(key);
                
                if length(symbolicPath) < 2
                    continue;
                end
                
                % Crea tabella entry/exit CP per questo path
                entryExitTable = obj.createEntryExitCPTable(symbolicPath);
                
                if ~isempty(entryExitTable)
                    obj.cpRoutingTable(key) = entryExitTable;
                end
            end
            
            if obj.debugMode
                fprintf('Routing geometrico: %d path con entry/exit CP\n', length(keys(obj.cpRoutingTable)));
            end
        end
        
        function entryExitTable = createEntryExitCPTable(obj, symbolicPath)
            % Crea tabella con entry/exit CP per ogni nodo nel path simbolico
            
            nNodes = length(symbolicPath);
            entryExitTable = struct();
            entryExitTable.nodes = symbolicPath;
            entryExitTable.nodeNames = cell(1, nNodes);
            entryExitTable.entryCP = zeros(1, nNodes);
            entryExitTable.exitCP = zeros(1, nNodes);
            entryExitTable.entryCP_global = zeros(1, nNodes);
            entryExitTable.exitCP_global = zeros(1, nNodes);
            
            for i = 1:nNodes
                currentNode = symbolicPath(i);
                cpData = obj.controlPoints(currentNode);
                
                entryExitTable.nodeNames{i} = obj.getNodeName(currentNode);
                
                if i == 1
                    % PRIMO NODO: solo exit CP
                    if nNodes > 1
                        nextNode = symbolicPath(i+1);
                        exitCP_local = obj.findCPConnectionToNode(currentNode, nextNode);
                        entryExitTable.entryCP(i) = 0; % N/A
                        entryExitTable.exitCP(i) = exitCP_local;
                        entryExitTable.entryCP_global(i) = 0; % N/A
                        entryExitTable.exitCP_global(i) = cpData.globalIndices(exitCP_local);
                    else
                        entryExitTable.entryCP(i) = 1;
                        entryExitTable.exitCP(i) = 1;
                        entryExitTable.entryCP_global(i) = cpData.globalIndices(1);
                        entryExitTable.exitCP_global(i) = cpData.globalIndices(1);
                    end
                    
                elseif i == nNodes
                    % ULTIMO NODO: solo entry CP
                    prevNode = symbolicPath(i-1);
                    entryCP_local = obj.findCPConnectionToNode(currentNode, prevNode);
                    entryExitTable.entryCP(i) = entryCP_local;
                    entryExitTable.exitCP(i) = 0; % N/A
                    entryExitTable.entryCP_global(i) = cpData.globalIndices(entryCP_local);
                    entryExitTable.exitCP_global(i) = 0; % N/A
                    
                else
                    % NODI INTERMEDI: entry + exit CP
                    prevNode = symbolicPath(i-1);
                    nextNode = symbolicPath(i+1);
                    
                    entryCP_local = obj.findCPConnectionToNode(currentNode, prevNode);
                    exitCP_local = obj.findCPConnectionToNode(currentNode, nextNode);
                    
                    entryExitTable.entryCP(i) = entryCP_local;
                    entryExitTable.exitCP(i) = exitCP_local;
                    entryExitTable.entryCP_global(i) = cpData.globalIndices(entryCP_local);
                    entryExitTable.exitCP_global(i) = cpData.globalIndices(exitCP_local);
                end
            end
        end
        
        function cpIdx = findCPConnectionToNode(obj, nodeIdx, targetNodeIdx)
            % Trova il CP di nodeIdx che è connesso a targetNodeIdx
            % basandosi sulla tabella delle connessioni definita manualmente
            
            cpData = obj.controlPoints(nodeIdx);
            
            % Per nodi WP: sempre CP 1
            if strcmp(cpData.nodeType, 'P') || contains(cpData.nodeType, 'W')
                cpIdx = 1;
                return;
            end
            
            % Per nodi T/X: cerca nella matrice delle connessioni geometriche
            % quale CP di questo nodo è collegato a un CP del nodo target
            
            for cp_local = 1:cpData.nCPs
                cp_global = cpData.globalIndices(cp_local);
                
                % Controlla se questo CP è collegato a qualche CP del nodo target
                connectedCPs = find(obj.geometricEdges(cp_global, :) > 0);
                
                for connected_cp = connectedCPs
                    if obj.cpNodeMap(connected_cp) == targetNodeIdx
                        cpIdx = cp_local;
                        return;
                    end
                end
            end
            
            % Fallback: usa geometria
            cpIdx = obj.findCPByGeometry(nodeIdx, targetNodeIdx);
        end
        
        function cpIdx = findCPByGeometry(obj, nodeIdx, targetNodeIdx)
            % Fallback: trova CP più vicino geometricamente
            cpData = obj.controlPoints(nodeIdx);
            nodePos = obj.nodes(nodeIdx).position;
            targetPos = obj.nodes(targetNodeIdx).position;
            direction = atan2(targetPos(2) - nodePos(2), targetPos(1) - nodePos(1));
            
            minAngularDist = inf;
            bestCP = 1;
            
            for i = 1:cpData.nCPs
                cpPos = cpData.positions(i, :);
                cpDirection = atan2(cpPos(2) - nodePos(2), cpPos(1) - nodePos(1));
                angularDist = abs(angdiff(direction, cpDirection));
                
                if angularDist < minAngularDist
                    minAngularDist = angularDist;
                    bestCP = i;
                end
            end
            
            cpIdx = bestCP;
        end
        
        % === METODO PER OTTENERE CP DA TABELLA PRECALCOLATA ===
        function [entryCP, exitCP] = getCPsFromPrecomputedTable(obj, agvId, targetNode)
            % Usa le chiavi precalcolate invece di calcolare dinamicamente

            agv = obj.AGVs(agvId);
            currentNode = agv.logicalNode;

            % Determina quale chiave usare
            routeKey = obj.getCurrentRouteKey(agvId, targetNode);

            if obj.cpRoutingTable.isKey(routeKey)
                routeData = obj.cpRoutingTable(routeKey);

                % Trova indici dei nodi nel path precalcolato
                currentNodeIndex = find(routeData.nodes == currentNode, 1);
                targetNodeIndex = find(routeData.nodes == targetNode, 1);

                if ~isempty(currentNodeIndex) && ~isempty(targetNodeIndex)
                    % Exit CP dal nodo corrente
                    if currentNodeIndex < length(routeData.nodes)
                        exitCP = routeData.exitCP(currentNodeIndex);
                    else
                        exitCP = 1; % Default per nodo finale
                    end

                    % Entry CP nel nodo target
                    entryCP = routeData.entryCP(targetNodeIndex);
                else
                    % Fallback geometrico
                    [entryCP, exitCP] = obj.calculateCPsGeometrically(currentNode, targetNode);
                end
            else
                % Fallback geometrico
                [entryCP, exitCP] = obj.calculateCPsGeometrically(currentNode, targetNode);
            end
        end

        function routeKey = getCurrentRouteKey(obj, agvId, targetNode)
            % Determina quale chiave usare basandosi sullo stato dell'AGV

            agv = obj.AGVs(agvId);

            if isempty(agv.routeKeys) || agv.currentSegmentIndex == 0
                % Fallback: calcola chiave dinamicamente
                routeKey = sprintf('%d_%d', agv.logicalNode, targetNode);
                return;
            end

            % Se abbiamo solo una chiave (es. movimento verso parcheggio)
            if length(agv.routeKeys) == 1
                routeKey = agv.routeKeys{1};
                return;
            end

            % Se abbiamo multiple chiavi, determina quale usare
            if agv.currentSegmentIndex <= length(agv.routeKeys)
                routeKey = agv.routeKeys{agv.currentSegmentIndex};
            else
                % Usa ultima chiave disponibile
                routeKey = agv.routeKeys{end};
            end
        end
        
        function [entryCP, exitCP] = calculateCPsGeometrically(obj, fromNode, toNode)
            % Fallback geometrico per route non precalcolate
            entryCP = obj.findCPConnectionToNode(toNode, fromNode);
            exitCP = obj.findCPConnectionToNode(fromNode, toNode);
        end

        % === FUNZIONI AUSILIARIE ===
        function name = getNodeName(obj, nodeIdx)
            if obj.indexNameMap.isKey(nodeIdx)
                name = obj.indexNameMap(nodeIdx);
            else
                name = sprintf('N%d', nodeIdx);
            end
        end
        
        function printCPSummary(obj)
            %fprintf('\n=== SOMMARIO CONTROL POINTS ===\n');
            nodeIndices = keys(obj.controlPoints);
            for i = 1:length(nodeIndices)
                nodeIdx = nodeIndices{i};
                cpData = obj.controlPoints(nodeIdx);
                nodeName = obj.getNodeName(nodeIdx);
                
                %fprintf('%s (%s): %d CP, indices %d-%d\n', ...
                %    nodeName, cpData.nodeType, cpData.nCPs, ...
                %    cpData.globalIndices(1), cpData.globalIndices(end));
            end
            %fprintf('Totale CP: %d\n\n', size(obj.cpPositions, 1));
        end
        
        % === VISUALIZZAZIONE ===
        function visualizeCompleteLayout(obj)
            % Debug dettagliato prima della visualizzazione
            %fprintf('\n=== DEBUG VISUALIZZAZIONE ===\n');
            %fprintf('cpPositions size: [%d x %d]\n', length(obj.cpPositions), width(obj.cpPositions));
            %fprintf('cpPositions class: %s\n', class(obj.cpPositions));
            
            if isempty(obj.cpPositions)
                %fprintf('ERRORE: cpPositions è empty!\n');
                return;
            end
            
            if ~isnumeric(obj.cpPositions)
                %fprintf('ERRORE: cpPositions non è numerico!\n');
                return;
            end
            
            % Visualizza layer simbolico
            obj.visualizeSymbolicLayer();
            
            % Visualizza layer geometrico solo se i dati sono validi
            if length(obj.cpPositions) > 0
                obj.visualizeGeometricLayer();
                obj.visualizeCombinedLayers();
            else
                %fprintf('Saltando visualizzazione geometrica: nessun CP valido\n');
            end
        end
        
        function visualizeSymbolicLayer(obj)
            obj.figHandles.symbolic = figure('Name', 'Layer Simbolico', 'Position', [100, 100, 800, 600]);
            hold on;
            
            % Disegna archi simbolici
            [row, col] = find(triu(obj.edges > 0));
            for i = 1:length(row)
                pos1 = obj.nodePositions(row(i), :);
                pos2 = obj.nodePositions(col(i), :);
                plot([pos1(1), pos2(1)], [pos1(2), pos2(2)], 'k-', 'LineWidth', 1);
            end
            
            % Disegna nodi simbolici
            for i = 1:length(obj.nodes)
                node = obj.nodes(i);
                pos = node.position;
                nodeType = node.type;
                nodeName = obj.getNodeName(i);
                
                [color, marker, markerSize] = obj.getNodeVisualizationParams(nodeType);
                
                plot(pos(1), pos(2), marker, 'MarkerSize', markerSize, ...
                    'MarkerFaceColor', color, 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
                
                text(pos(1)+2, pos(2)+2, nodeName, 'FontSize', 10, 'Interpreter', 'latex');
            end
            
            xlabel('X (m)', 'Interpreter', 'latex');
            ylabel('Y (m)', 'Interpreter', 'latex');
            title('Layer Simbolico - Nodi e Connessioni', 'Interpreter', 'latex');
            grid on;
            axis equal;
            
            % Legenda
            obj.addSymbolicLayerLegend();
        end
        
        function visualizeGeometricLayer(obj)
            % Verifica che i dati siano inizializzati
            if isempty(obj.cpPositions)
                error('cpPositions è vuoto! Problema nell''inizializzazione del layer geometrico.');
            end
            if isempty(obj.geometricEdges)
                error('geometricEdges è vuoto! Problema nel calcolo delle connessioni geometriche.');
            end
            
            obj.figHandles.geometric = figure('Name', 'Layer Geometrico', 'Position', [920, 100, 800, 600]);
            hold on;
            
            % Disegna connessioni geometriche
            [cpRow, cpCol] = find(triu(obj.geometricEdges > 0));
            for i = 1:length(cpRow)
                pos1 = obj.cpPositions(cpRow(i), :);
                pos2 = obj.cpPositions(cpCol(i), :);
                
                % Distingui connessioni intra/inter nodo
                node1 = obj.cpNodeMap(cpRow(i));
                node2 = obj.cpNodeMap(cpCol(i));
                
                if node1 == node2
                    % Connessione intra-nodo
                    plot([pos1(1), pos2(1)], [pos1(2), pos2(2)], 'g-', 'LineWidth', 1.5);
                else
                    % Connessione inter-nodo
                    plot([pos1(1), pos2(1)], [pos1(2), pos2(2)], 'b-', 'LineWidth', 1);
                end
            end
            
            % Disegna CP
            nCPs = length(obj.cpPositions);
            for i = 1:nCPs
                pos = obj.cpPositions(i, :);
                nodeIdx = obj.cpNodeMap(i);
                nodeType = obj.nodes(nodeIdx).type;
                
                color = obj.getCPColor(nodeType);
                plot(pos(1), pos(2), 'o', 'MarkerSize', 6, ...
                    'MarkerFaceColor', color, 'MarkerEdgeColor', 'k');
                
                text(pos(1)+1, pos(2)+1, sprintf('%d', i), 'FontSize', 6);
            end
            
            % Mostra centri nodi simbolici come riferimento
            for i = 1:length(obj.nodes)
                pos = obj.nodes(i).position;
                plot(pos(1), pos(2), 'kx', 'MarkerSize', 8, 'LineWidth', 2);
            end
            
            xlabel('X (m)', 'Interpreter', 'latex');
            ylabel('Y (m)', 'Interpreter', 'latex');
            title('Layer Geometrico - Control Points e Connessioni', 'Interpreter', 'latex');
            grid on;
            axis equal;
            
            legend({'Conn. Intra-node', 'Conn. Inter-node', 'CP', 'Node Center'}, ...
                'Location', 'best', 'Interpreter', 'latex');
        end
        
        function visualizeCombinedLayers(obj)
            % Verifica che i dati siano inizializzati
            if isempty(obj.cpPositions)
                error('cpPositions è vuoto! Problema nell''inizializzazione del layer geometrico.');
            end
            
            obj.figHandles.combined = figure('Name', 'Layout Completo', 'Position', [100, 750, 1200, 600]);
            hold on;
            
            % Layer simbolico (linee sottili)
            [row, col] = find(triu(obj.edges > 0));
            for i = 1:length(row)
                pos1 = obj.nodePositions(row(i), :);
                pos2 = obj.nodePositions(col(i), :);
                plot([pos1(1), pos2(1)], [pos1(2), pos2(2)], 'k--', 'LineWidth', 0.5, 'Color', [0.7 0.7 0.7]);
            end
            
            % Layer geometrico
            [cpRow, cpCol] = find(triu(obj.geometricEdges > 0));
            for i = 1:length(cpRow)
                pos1 = obj.cpPositions(cpRow(i), :);
                pos2 = obj.cpPositions(cpCol(i), :);
                
                node1 = obj.cpNodeMap(cpRow(i));
                node2 = obj.cpNodeMap(cpCol(i));
                
                if node1 == node2
                    plot([pos1(1), pos2(1)], [pos1(2), pos2(2)], 'g-', 'LineWidth', 1);
                else
                    plot([pos1(1), pos2(1)], [pos1(2), pos2(2)], 'b-', 'LineWidth', 0.8);
                end
            end
            
            % Nodi simbolici
            for i = 1:length(obj.nodes)
                node = obj.nodes(i);
                pos = node.position;
                nodeType = node.type;
                nodeName = obj.getNodeName(i);
                
                [color, marker, markerSize] = obj.getNodeVisualizationParams(nodeType);
                
                plot(pos(1), pos(2), marker, 'MarkerSize', markerSize, ...
                    'MarkerFaceColor', color, 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
                
                text(pos(1)+3, pos(2)+3, nodeName, 'FontSize', 10, 'FontWeight', 'bold', 'Interpreter', 'latex');
            end
            
            % CP
            nCPs = length(obj.cpPositions);
            for i = 1:nCPs
                pos = obj.cpPositions(i, :);
                nodeIdx = obj.cpNodeMap(i);
                nodeType = obj.nodes(nodeIdx).type;
                
                color = obj.getCPColor(nodeType);
                plot(pos(1), pos(2), 'o', 'MarkerSize', 4, ...
                    'MarkerFaceColor', color, 'MarkerEdgeColor', 'k');
            end
            
            xlabel('X (m)', 'Interpreter', 'latex');
            ylabel('Y (m)', 'Interpreter', 'latex');
            title('Layout Completo - Layer Simbolico + Geometrico', 'Interpreter', 'latex');
            grid on;
            axis equal;
            
            obj.addCombinedLayerLegend();
        end
        
        function [color, marker, markerSize] = getNodeVisualizationParams(obj, nodeType)
            switch nodeType
                case 'P'
                    color = 'blue'; marker = 's'; markerSize = 10;
                case 'W_pickup'
                    color = 'green'; marker = 'o'; markerSize = 10;
                case 'W_dropoff'
                    color = 'red'; marker = 'o'; markerSize = 10;
                case 'T'
                    color = [0.5 0.5 0.5]; marker = '^'; markerSize = 8;
                case 'X'
                    color = [1 0.5 0]; marker = 'p'; markerSize = 12;
                otherwise
                    color = 'black'; marker = 'o'; markerSize = 6;
            end
        end
        
        function color = getCPColor(obj, nodeType)
            switch nodeType
                case 'P'
                    color = 'cyan';
                case 'W_pickup'
                    color = [0.5 1 0.5];  % light green
                case 'W_dropoff'
                    color = [1 0.7 0.7];  % light red/pink
                case 'T'
                    color = 'yellow';
                case 'X'
                    color = [1 0.5 0];    % orange
                otherwise
                    color = 'white';
            end
        end
        
        function addSymbolicLayerLegend(obj)
            % Crea handle invisibili per la legenda
            h = [];
            h(1) = plot(nan, nan, 's', 'MarkerSize', 10, 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'k');
            h(2) = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'green', 'MarkerEdgeColor', 'k');
            h(3) = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'k');
            h(4) = plot(nan, nan, '^', 'MarkerSize', 8, 'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerEdgeColor', 'k');
            h(5) = plot(nan, nan, 'p', 'MarkerSize', 12, 'MarkerFaceColor', [1 0.5 0], 'MarkerEdgeColor', 'k');
            
            legend(h, {'Parking Station (P)', 'Pickup Station (W)', 'Dropoff Station (W)', 'T Cross',  'X Cross'}, ...
                'Location', 'bestoutside', 'Interpreter', 'latex');
        end
        
        function addCombinedLayerLegend(obj)
            h = [];
            h(1) = plot(nan, nan, 's', 'MarkerSize', 10, 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'k');
            h(2) = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'green', 'MarkerEdgeColor', 'k');
            h(3) = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'red', 'MarkerEdgeColor', 'k');
            h(4) = plot(nan, nan, '^', 'MarkerSize', 8, 'MarkerFaceColor', [0.5 0.5 0.5], 'MarkerEdgeColor', 'k');
            h(5) = plot(nan, nan, 'p', 'MarkerSize', 12, 'MarkerFaceColor', [1 0.5 0], 'MarkerEdgeColor', 'k');
            h(6) = plot(nan, nan, 'o', 'MarkerSize', 4, 'MarkerFaceColor', 'cyan', 'MarkerEdgeColor', 'k');
            h(7) = plot(nan, nan, 'g-', 'LineWidth', 1);
            h(8) = plot(nan, nan, 'b-', 'LineWidth', 0.8);
            
            legend(h, {'Parking Station', 'Pickup Station', 'Dropoff Station', 'T Cross', 'X Cross', ...
                'Control Point', 'Conn. Intra-node', 'Conn. Inter-node'}, ...
                'Location', 'bestoutside', 'Interpreter', 'latex');
        end

        % === INIZIALIZZAZIONE COMPONENTI RL ===
        function initializeRLComponents(obj)
            obj.top5TaskQueue = [];
            obj.pendingRLDecisions = [];
            obj.rlDecisionTimestamps = [];

            % CORREZIONE: Inizializza correttamente state history
            obj.agvStateHistory = cell(obj.nAGV, 1);
            for i = 1:obj.nAGV
                obj.agvStateHistory{i} = struct.empty(); % Array di struct vuoto
            end

            obj.agvStateTimes = repmat(struct(...
                'Idle', 0, 'Resuming', 0, 'Waiting', 0, 'Resolving', 0, ...
                'lastTransition', 0, 'currentState', 0), obj.nAGV, 1);

            for i = 1:obj.nAGV
                obj.agvStateTimes(i).lastTransition = obj.currentTime;
                obj.agvStateTimes(i).currentState = 0; % idle
            end

            obj.resetThroughputTracking();
            obj.logMessage('Componenti RL inizializzati');
        end

        function isDeadlocked = isDeadlock(obj)
            if obj.nAGV == 0
                isDeadlocked = false;
                return;
            end

            % Conta AGV in stato WAITING (state == 2) E fisicamente fermi
            waitingAndStillCount = 0;
            for agvId = 1:obj.nAGV
                if obj.AGVs(agvId).state == 2 && ~obj.AGVs(agvId).isMoving
                    waitingAndStillCount = waitingAndStillCount + 1;
                end
            end

            % Deadlock solo se TUTTI sono WAITING e NESSUNO si muove fisicamente
            isDeadlocked = (waitingAndStillCount == obj.nAGV);

            if isDeadlocked && obj.verboseLogging
                fprintf('⚠️ Tutti i %d AGV: WAITING e fermi\n', obj.nAGV);
            end
        end

        function setOperatingMode(obj, mode, environmentHandle)
            if nargin < 3, environmentHandle = []; end

            validModes = {'IDRR', 'RL_TRAINING', 'RL_TESTING', 'PRIM_MS', 'PRIM_MM'};            
            if ~ismember(mode, validModes)
                error('Modalità non valida. Scegliere: %s', strjoin(validModes, ', '));
            end

            previousMode = obj.operatingMode;
            obj.operatingMode = mode;

            if ~isempty(environmentHandle)
                obj.connectToEnvironment(environmentHandle);
            end

            obj.logMessage(sprintf('Modalità: %s → %s', previousMode, mode));
        end

        function connectToEnvironment(obj, environmentHandle)
            obj.environmentInterface = environmentHandle;
            obj.logMessage('Connesso a AGV_Environment');
        end
      
        % === INIZIALIZZAZIONE STATI SISTEMA ===
        function initializeSystemStates(obj)
            nNodes = length(obj.nodes);
            nCPs = size(obj.cpPositions, 1);
            
            obj.nodeStates = zeros(nNodes, 1);
            obj.nodeOccupants = cell(nNodes, 1);
            for i = 1:nNodes
                obj.nodeOccupants{i} = [];
            end
            
            obj.cpStates = zeros(nCPs, 1);
            obj.cpOccupants = zeros(nCPs, 1);
        end
        
        function initializeAGVs(obj)
            % Inizializza AGV con posizioni randomizzate per training
            parkingNodes = find(strcmp(obj.nodeTypes, 'P'));
            obj.AGVs = repmat(struct(), obj.nAGV, 1);

            % Seleziona posizioni iniziali based su modalità
            switch obj.operatingMode
                case {'IDRR', 'RL_TESTING', 'PRIM_MS', 'PRIM_MM'}
                    % Posizioni deterministiche (originale)
                    selectedParking = zeros(obj.nAGV, 1);
                    for i = 1:obj.nAGV
                        selectedParking(i) = parkingNodes(mod(i-1, length(parkingNodes)) + 1);
                    end
                    obj.logMessage('Posizioni AGV deterministiche');

                case 'RL_TRAINING'
                    % Posizioni casuali per training
                    if obj.nAGV > length(parkingNodes)
                        error('Troppi AGV per i parcheggi disponibili: %d AGV, %d parcheggi', ...
                            obj.nAGV, length(parkingNodes));
                    end

                    % Seleziona casualmente parcheggi unici
                    selectedParking = parkingNodes(randperm(length(parkingNodes), obj.nAGV));
                    obj.logMessage(sprintf('Posizioni AGV randomizzate: %s', ...
                        mat2str(selectedParking)));
            end

            % Inizializza AGV con posizioni selezionate
            for i = 1:obj.nAGV
                parkingIdx = selectedParking(i);

                obj.AGVs(i).id = i;
                obj.AGVs(i).logicalNode = parkingIdx;
                obj.AGVs(i).logicalCP = 1;
                obj.AGVs(i).isMoving = false;
                obj.AGVs(i).arrivalTime = 0;
                obj.AGVs(i).state = 0; % idle
                obj.AGVs(i).task = [];
                obj.AGVs(i).residualRoute = [];
                obj.AGVs(i).sharedRoute = [];
                obj.AGVs(i).routeKeys = {};
                obj.AGVs(i).currentSegmentIndex = 0;
                obj.AGVs(i).finalTarget = [];
                obj.AGVs(i).tasksCompleted = 0;
                obj.AGVs(i).totalDistance = 0;
                obj.AGVs(i).taskDistance = 0;
                obj.AGVs(i).waitingTime = 0;
                obj.AGVs(i).totalPlannedDistance = 0;
                obj.AGVs(i).currentTaskStateTimes = struct(...
                'Idle', 0, 'Resuming', 0, 'Waiting', 0, 'Resolving', 0, ...
                'taskStartTime', 0);
                obj.AGVs(i).maneuverCounts = struct('res1', 0, 'res2', 0);
                obj.AGVs(i).unexecutedTask = [];    % Coda Qu_i per PRIM (Def. 5.1, Cap. 5)

                % Prenota CP e aggiorna stati
                if obj.controlPoints.isKey(parkingIdx)
                    cpData = obj.controlPoints(parkingIdx);
                    globalCPIdx = cpData.globalIndices(1);
                    obj.cpStates(globalCPIdx) = 1;
                    obj.cpOccupants(globalCPIdx) = i;
                    obj.nodeStates(parkingIdx) = obj.nodeStates(parkingIdx) + 1;
                    obj.nodeOccupants{parkingIdx} = [obj.nodeOccupants{parkingIdx}, i];
                end
            end

            obj.logMessage(sprintf('AGV inizializzati: %d in modalità %s', obj.nAGV, obj.operatingMode));
        end
        
        function initializeTaskManagement(obj)
            obj.taskQueue = [];
            obj.completedTasks = [];
            obj.pendingManeuvers = []; % Inizializza array manovre pendenti
            obj.activeManeuvers = [];
            obj.primAuctionPool = [];          % Pool asta PRIM (TA in Algorithm 3)
            obj.preGeneratedTasks = [];        % Lista task pre-generati (sovrascritto da loadPreGeneratedTasks)
        end

        function loadPreGeneratedTasks(obj, fixedTasks, initialCount)
            % Carica task pre-generati per TUTTE le modalità di testing.
            % I primi initialCount task sono disponibili subito, il resto
            % viene iniettato ogni taskInjectionInterval secondi.
            % Garantisce condizioni identiche di arrivo task tra modalità.
            
            obj.preGeneratedTasks = fixedTasks;
            obj.preGenTaskIndex = initialCount + 1;  % Prossimo task da iniettare
            obj.lastInjectionTime = 0;
            obj.tasksGenerated = initialCount;       % Aggiorna contatore: 12 task già disponibili
            obj.taskCounter = length(fixedTasks);    % Evita conflitti sequentialId con generateRandomTask
            
            % Imposta creationTime e registra entry times per i task iniziali
            for i = 1:initialCount
                fixedTasks(i).creationTime = 0;
                obj.taskPoolEntryTimes(fixedTasks(i).sequentialId) = 0;
            end
            
            % Destinazione dei task iniziali dipende dalla modalità:
            % PRIM → pool di asta, FIFO/MAPPO → coda globale
            if strcmp(obj.operatingMode, 'PRIM_MS') || strcmp(obj.operatingMode, 'PRIM_MM')
                obj.primAuctionPool = fixedTasks(1:initialCount);
            else
                % IDRR (FIFO) / RL_TESTING (MAPPO)
                obj.taskQueue = fixedTasks(1:initialCount);
            end
            
            if obj.debugMode
                fprintf('📦 Task caricati: %d iniziali, %d da iniettare (ogni %.1fs) - Modalità: %s\n', ...
                    initialCount, length(fixedTasks) - initialCount, ...
                    obj.taskInjectionInterval, obj.operatingMode);
            end
        end
        
        function initializeSharedRoutes(obj)
            obj.sharedRoutes = containers.Map();
        end
        
        function initializeMetrics(obj)
            obj.metrics = struct();
            obj.metrics.conflictCounts = struct('headon', 0, 'intersection', 0, 'pursuit', 0, 'loop', 0);
            obj.metrics.maneuverCounts = struct('res1', 0, 'res2', 0);
            obj.metrics.productivity = 0;
            obj.activeConflicts = containers.Map('KeyType', 'char', 'ValueType', 'double');

            % Inizializza tracking task delay nel pool
            obj.taskPoolEntryTimes = containers.Map('KeyType', 'double', 'ValueType', 'double');
            obj.taskDelaysInPool = [];

            % Inizializza contatori pickup station
            pickupNodes = find(strcmp(obj.nodeTypes, 'W_pickup'));
            obj.pickupStationCounts = containers.Map('KeyType', 'double', 'ValueType', 'double');
            for i = 1:length(pickupNodes)
                obj.pickupStationCounts(pickupNodes(i)) = 0;
            end
        end

        function loadEventDropoffSequences(obj, dropoffSeqs, pickupNodes)
            % Carica le sequenze pre-generate e inietta i task iniziali (1 per stazione).
            obj.eventDrivenMode        = true;
            obj.eventDropoffSequences  = dropoffSeqs;
            obj.eventPickupNodes       = pickupNodes(:);
            obj.eventPickupCounters    = containers.Map('KeyType','int32','ValueType','int32');

            % Bootstrap: 1 task iniziale per ogni stazione pickup (usa il primo dropoff in sequenza)
            for p = 1:length(pickupNodes)
                pNode = int32(pickupNodes(p));
                obj.eventPickupCounters(pNode) = 1;   % Counter parta da 1 (primo già iniettato)

                dropoffNode = dropoffSeqs(p, 1);
                obj.taskCounter = obj.taskCounter + 1;
                task = struct('sequentialId', obj.taskCounter, ...
                    'pickup',       pickupNodes(p), ...
                    'dropoff',      dropoffNode, ...
                    'taskTypeId',   obj.computeTaskTypeId(pickupNodes(p), dropoffNode), ...
                    'creationTime', 0, ...
                    'assignmentTime', -1, ...
                    'assignedAGV',  0, ...
                    'totalDistance', 0);

                obj.taskPoolEntryTimes(task.sequentialId) = 0;
                obj.tasksGenerated = obj.tasksGenerated + 1;

                if strcmp(obj.operatingMode,'PRIM_MS') || strcmp(obj.operatingMode,'PRIM_MM')
                    obj.primAuctionPool = [obj.primAuctionPool, task];
                else
                    obj.taskQueue = [obj.taskQueue, task];
                end
            end
            fprintf('✅ Event-driven mode: %d task iniziali iniettati (1 per stazione pickup)\n', ...
                length(pickupNodes));
        end

        function injectEventDrivenTask(obj, pickupNode)
            % Chiamato ogni volta che un AGV parte da un nodo pickup.
            key = int32(pickupNode);
            if ~obj.eventPickupCounters.isKey(key), return; end

            counter = obj.eventPickupCounters(key) + 1;
            pickupIdx = find(obj.eventPickupNodes == pickupNode, 1);

            if isempty(pickupIdx) || counter > size(obj.eventDropoffSequences, 2)
                if obj.debugMode
                    fprintf('⚠️  Event-driven: sequenza esaurita per pickup %d\n', pickupNode);
                end
                return;
            end

            dropoffNode = obj.eventDropoffSequences(pickupIdx, counter);
            obj.eventPickupCounters(key) = counter;

            obj.taskCounter = obj.taskCounter + 1;
            task = struct('sequentialId', obj.taskCounter, ...
                'pickup',       pickupNode, ...
                'dropoff',      dropoffNode, ...
                'taskTypeId',   obj.computeTaskTypeId(pickupNode, dropoffNode), ...
                'creationTime', obj.currentTime, ...
                'assignmentTime', -1, ...
                'assignedAGV',  0, ...
                'totalDistance', 0);

            obj.taskPoolEntryTimes(task.sequentialId) = obj.currentTime;
            obj.tasksGenerated = obj.tasksGenerated + 1;

            if strcmp(obj.operatingMode,'PRIM_MS') || strcmp(obj.operatingMode,'PRIM_MM')
                obj.primAuctionPool = [obj.primAuctionPool, task];
            else
                obj.taskQueue = [obj.taskQueue, task];
            end

            if obj.debugMode
                fprintf('📦 Event-task %d: pickup=%d dropoff=%d (evento #%d per stazione)\n', ...
                    task.sequentialId, pickupNode, dropoffNode, counter);
            end
        end

        % === SIMULAZIONE PRINCIPALE ===
        function runSimulation(obj)
            %fprintf('Inizio simulazione IDRR\n');

            % Avvia visualizzazione se abilitata
            if obj.visualizationEnabled
                obj.startVisualization();
            end

            taskInterval = 1 / obj.taskGenerationRate;

            while obj.currentTime < obj.maxTime && ~obj.isSimulationComplete()
                % ===== ALGORITHM 1: TASK AND ROUTE PLANNING =====
                obj.executeTaskAndRouteManagement(taskInterval);

                % ===== ALGORITHM 2: TRAFFIC CONTROL =====
                obj.executeTrafficControl();

                % ===== AGGIORNAMENTO MANOVRE ATTIVE =====
                obj.updateActiveManeuvers();

                % ===== PHYSICAL MOVEMENTS UPDATE =====
                obj.updatePhysicalMovements();

                % ===== CONTROLLO DEADLOCK =====
                % Check ogni 10 timesteps (= ogni 1.0s simulato).
                % Il counter viene incrementato se TUTTI gli AGV sono in Waiting
                % e nessuno si muove fisicamente; resettato appena uno si sblocca.
                % Soglia: 50 check consecutivi = 50 secondi simulati di stallo.
                if obj.enableDeadlockDetection
                    if mod(round(obj.currentTime / obj.timeStep), 10) == 0
                        % Deadlock reale: tutti waiting, nessun moto E
                        % nessuna manovra pending/active che possa sbloccare.
                        % La seconda condizione evita falsi positivi quando
                        % una Res2 pending è pronta ma non ancora promossa.
                        noPending = isempty(obj.pendingManeuvers) && ...
                                    isempty(obj.activeManeuvers);
                        if obj.isDeadlock() && noPending
                            obj.deadlockCounter = obj.deadlockCounter + 1;
                        else
                            obj.deadlockCounter = 0;
                        end
                        if obj.deadlockCounter >= 50
                            stalledSeconds = obj.deadlockCounter * obj.timeStep * 10;
                            error('AGV_DEADLOCK:AllWaiting', ...
                                'Deadlock [%s]: tutti i %d AGV bloccati per %.0fs al t=%.1fs', ...
                                obj.operatingMode, obj.nAGV, stalledSeconds, obj.currentTime);
                        end
                    end
                end

                % ===== METRICHE E TIME STEP =====
                obj.recordMetrics();
                obj.currentTime = obj.currentTime + obj.timeStep;

                % ===== AGGIORNAMENTO VISUALIZZAZIONE =====
                obj.updateVisualization();

                if mod(obj.currentTime, 100) < obj.timeStep
                    %obj.printSimulationStatus();
                end
            end
                 
            obj.computeFinalMetrics();
            obj.collectEpisodeMetrics()
            
            %fprintf('Simulazione completata con successo\n');

            % Mantieni visualizzazione aperta alla fine
            if obj.visualizationEnabled
                %fprintf('💡 Visualizzazione rimane aperta. Usa obj.closeVisualization() per chiudere.\n');
            end
        end
        
        % === ALGORITHM 1: TASK AND ROUTE PLANNING ===
function executeTaskAndRouteManagement(obj, taskInterval)
            % Line 2: Update assignment list A
            % === INIEZIONE TASK ===
            if strcmp(obj.operatingMode, 'RL_TRAINING')
                % RL_TRAINING: generazione random originale (necessaria per esplorazione)
                if (obj.currentTime - obj.lastTaskTime) >= taskInterval && obj.tasksGenerated < obj.maxTasks
                    obj.generateRandomTask();
                    obj.tasksGenerated = obj.tasksGenerated + 1;
                    obj.lastTaskTime = obj.currentTime;

                    if obj.debugMode
                        fprintf('Task generato al tempo %.1f (totale: %d/%d)\n', ...
                            obj.currentTime, obj.tasksGenerated, obj.maxTasks);
                    end
                end
            else
                % Modalità testing: iniezione uniforme da preGeneratedTasks
                if ~isempty(obj.preGeneratedTasks)
                    % Inietta 1 task ogni taskInjectionInterval secondi
                    if (obj.currentTime - obj.lastInjectionTime) >= obj.taskInjectionInterval && ...
                       obj.preGenTaskIndex <= length(obj.preGeneratedTasks)

                        task = obj.preGeneratedTasks(obj.preGenTaskIndex);
                        task.creationTime = obj.currentTime;
                        obj.taskPoolEntryTimes(task.sequentialId) = obj.currentTime;

                        % Destinazione: pool asta per PRIM, coda globale per FIFO/MAPPO
                        if strcmp(obj.operatingMode, 'PRIM_MS') || strcmp(obj.operatingMode, 'PRIM_MM')
                            obj.primAuctionPool = [obj.primAuctionPool, task];
                        else
                            % IDRR (FIFO) / RL_TESTING (MAPPO)
                            obj.taskQueue = [obj.taskQueue, task];
                        end

                        obj.preGenTaskIndex = obj.preGenTaskIndex + 1;
                        obj.tasksGenerated = obj.tasksGenerated + 1;
                        obj.lastInjectionTime = obj.currentTime;

                        if obj.debugMode
                            fprintf('⏰ Task %d iniettato a t=%.1fs (%d/%d)\n', ...
                                task.sequentialId, obj.currentTime, ...
                                obj.tasksGenerated, length(obj.preGeneratedTasks));
                        end
                    end
                else
                    % Fallback: generazione random (uso standalone IDRR senza pre-generazione)
                    if (obj.currentTime - obj.lastTaskTime) >= taskInterval && obj.tasksGenerated < obj.maxTasks
                        obj.generateRandomTask();
                        obj.tasksGenerated = obj.tasksGenerated + 1;
                        obj.lastTaskTime = obj.currentTime;
                    end
                end
            end

            % === BLOCCO PRIM: Asta + avvio task + parking (Algorithm 3, Cap. 5) ===
            if strcmp(obj.operatingMode, 'PRIM_MS') || strcmp(obj.operatingMode, 'PRIM_MM')

                % Algorithm 3, righe 3-8: Re-auctioning se ci sono task nuovi nel pool
                if ~isempty(obj.primAuctionPool)
                    % Riga 4: Raccoglie tutti i Qu_i (assegnati ma non eseguiti)
                    for i = 1:obj.nAGV
                        if ~isempty(obj.AGVs(i).unexecutedTask)
                            obj.primAuctionPool = [obj.primAuctionPool, obj.AGVs(i).unexecutedTask];
                            obj.AGVs(i).unexecutedTask = [];  % Righe 5-7: Set Qu_i = ∅
                        end
                    end

                    % Righe 9-23: Asta multi-round (assegna tutto il pool alle code degli AGV)
                    obj.runPRIMAuction();
                    obj.primAuctionPool = [];  % Pool svuotato dopo asta completa
                end

                % Avvia task per tutti gli AGV idle che hanno task nella coda Qu
                % (nessun break: ogni AGV pesca dalla propria coda indipendente)
                for agvId = 1:obj.nAGV
                    if obj.AGVs(agvId).state == 0 && ~isempty(obj.AGVs(agvId).unexecutedTask)
                        obj.startNextTaskForAGV(agvId);
                    end
                end

                % Manda al parcheggio AGV idle senza coda (un alla volta, come originale)
                for agvId = 1:obj.nAGV
                    agv = obj.AGVs(agvId);
                    if agv.state == 0 && isempty(agv.unexecutedTask)
                        currentNodeType = obj.nodes(agv.logicalNode).type;
                        if contains(currentNodeType, 'W')
                            route = obj.findRouteToNearestParkingDirect(agvId);
                            if ~isempty(route)
                                obj.AGVs(agvId).residualRoute = route;
                                obj.AGVs(agvId).totalPlannedDistance = obj.calculatePathDistance(route);
                                obj.changeAGVState(agvId, obj.AGVs(agvId).state, 2);
                                if obj.debugMode
                                    fprintf('🅿️ AGV %d → parcheggio %s\n', agvId, obj.getNodeName(route(end)));
                                end
                            end
                            break;  % Un AGV alla volta verso parking
                        end
                    end
                end

                return;  % Salta logica FIFO/RL sotto
            end
            
            % Lines 3-10: Assign tasks to idle AGVs
            obj.updateTop5TaskQueue();
            if ~isempty(obj.taskQueue)
                for agvId = 1:obj.nAGV
                    agv = obj.AGVs(agvId);
                    % Line 4: if (∃Ri ∈ R)&(Si = 0) then
                    if agv.state == 0 && ~isempty(obj.taskQueue)
                        if strcmp (obj.operatingMode,'IDRR')
                            obj.resetCurrentTaskStateTimes(agvId);
                            % Line 5: Find shortest route Ti to accomplish Aj
                            task = obj.taskQueue(1);
                            route = obj.planRouteForTaskSegmented(agvId, task);
                            
                            % Line 6: Send task Ti to Ri
                            obj.AGVs(agvId).task = task;
                            obj.AGVs(agvId).residualRoute = route;
                            obj.AGVs(agvId).task.assignmentTime = obj.currentTime;
                            
                            % Line 7: Remove Aj from A
                            obj.taskQueue(1) = [];
                            
                            % Line 8: Set Si = 2
                            obj.changeAGVState(agvId, obj.AGVs(agvId).state, 2); % waiting
    
                            if obj.debugMode
                                taskInfo = sprintf('Task %d: %s → %s', task.sequentialId, ...
                                    obj.getNodeName(task.pickup), obj.getNodeName(task.dropoff));
                                fprintf('✅ %s assegnato ad AGV %d\n', taskInfo, agvId);
                            end
                            
                            break;
                        else
                            obj.requestRLDecision(agvId);
                        end
                    end
                end
            end
            
            % Lines 11-17: Send idle AGVs to parking if no tasks
            if isempty(obj.taskQueue)
                for agvId = 1:obj.nAGV
                    agv = obj.AGVs(agvId);
                    currentNodeType = obj.nodes(agv.logicalNode).type;
                    
                    % Line 12: if (∃Ri ∈ R)&(Si = 0)&(Pi = W station) then
                    if agv.state == 0 && (contains(currentNodeType, 'W'))
                        % Line 13: Find shortest route Ti to nearest free P station
                        route = obj.findRouteToNearestParkingDirect(agvId);                        
                                    
                        if ~isempty(route)
                            % Line 14: Send task Ti to Ri
                            obj.AGVs(agvId).residualRoute = route;
                            totalDistance = obj.calculatePathDistance(route);
                            obj.AGVs(agvId).totalPlannedDistance = totalDistance;
                            
                            % Line 15: set Si = 2
                            obj.changeAGVState(agvId, obj.AGVs(agvId).state, 2); % waiting

                            if obj.debugMode
                                targetNodeName = obj.getNodeName(route(end));
                                fprintf('🅿️  AGV %d mandato al parcheggio %s\n', agvId, targetNodeName);
                            end
                           
                        end
                        break;
                    end
                end
            end
        end
        
        function executeTrafficControl(obj)
            % Raccogli AGV in waiting che non si muovono
            waitingAGVs = [];
            for agvId = 1:obj.nAGV
                agv = obj.AGVs(agvId);
                if agv.state == 2 && ~agv.isMoving
                    waitingAGVs = [waitingAGVs; agvId, agv.arrivalTime];
                end
            end

            % Processa in ordine di arrivalTime (chi è arrivato prima)
            if ~isempty(waitingAGVs)
                [~, sortIdx] = sort(waitingAGVs(:, 2));  % Ordina per arrivalTime

                for i = 1:length(sortIdx)
                    agvId = waitingAGVs(sortIdx(i), 1);
                    agv = obj.AGVs(agvId);

                    % Line 2: if (∃Ri ∈ R)&(Si = 2) then - SOLO AGV waiting e fermi
                    % [Qui metti tutto il resto del codice originale delle regole IDRR]

                    % Line 3: Read the travelling information TIi = (Nx, Ny)
                    [Nx, Ny] = obj.getTravellingInformation(agvId);

                    if isempty(Ny)
                        % Line 12: Set Si = 0
                        obj.handleTaskCompletion(agvId);
                        continue;
                    end

                    % if obj.AGVs(agvId).logicalNode == obj.AGVs(agvId).finalTarget && obj.currentTime >= obj.AGVs(agvId).arrivalTime
                    %     obj.handleTaskCompletion(agvId);
                    %     continue;
                    % end

                    % Line 4: Update residual route Li from TIi = (Nx, Ny)
                    obj.updateResidualRouteFromTI(agvId, Nx, Ny);

                    % Lines 5-7: Create shared routes with all other AGVs
                    for otherAgvId = 1:obj.nAGV
                        if otherAgvId ~= agvId
                            obj.createSharedRoute(agvId, otherAgvId);
                        end
                    end

                    % Line 8: Update whole shared route Σi
                    obj.updateWholeSharedRoute(agvId);

                    % Lines 9-10: Execute Traffic Rules
                    obj.executeTrafficRules(agvId);
                end
            end
        end
        
        % === FUNZIONI DI SUPPORTO PER ALGORITHM 1 E 2 ===
        % function generateRandomTask(obj)
        %     pickupNodes = find(strcmp(obj.nodeTypes, 'W_pickup'));
        %     dropoffNodes = find(strcmp(obj.nodeTypes, 'W_dropoff'));
        % 
        %     pickup = pickupNodes(randi(length(pickupNodes)));
        %     dropoff = dropoffNodes(randi(length(dropoffNodes)));
        % 
        %     task = struct();
        %     obj.taskCounter = obj.taskCounter + 1;
        %     task.sequentialId = obj.taskCounter;
        %     task.taskTypeId = obj.computeTaskTypeId(pickup, dropoff);
        %     task.pickup = pickup;
        %     task.dropoff = dropoff;
        %     task.creationTime = obj.currentTime;
        %     task.assignmentTime = -1;
        %     task.assignedAGV = 0;
        %     task.totalDistance = 0;
        % 
        %     obj.taskQueue = [obj.taskQueue, task];
        % end

        function generateRandomTask(obj)
            % Genera task random garantendo varietà (no duplicati in finestra di 7 task).
            % Se preloadedTaskList è valorizzata, usa quella lista in modo sequenziale
            % per garantire riproducibilità nel testing della baseline IDRR/FIFO.

            % === CASO 1: Lista pre-caricata (testing riproducibile IDRR baseline) ===
            if ~isempty(obj.preloadedTaskList) && ...
                    obj.preloadedTaskIdx < size(obj.preloadedTaskList, 1)
                obj.preloadedTaskIdx = obj.preloadedTaskIdx + 1;
                pickup  = obj.preloadedTaskList(obj.preloadedTaskIdx, 1);
                dropoff = obj.preloadedTaskList(obj.preloadedTaskIdx, 2);

                task = struct();
                obj.taskCounter    = obj.taskCounter + 1;
                task.sequentialId  = obj.taskCounter;
                task.taskTypeId    = obj.computeTaskTypeId(pickup, dropoff);
                task.pickup        = pickup;
                task.dropoff       = dropoff;
                task.creationTime  = obj.currentTime;
                task.assignmentTime = -1;
                task.assignedAGV   = 0;
                task.totalDistance = 0;
                obj.taskQueue = [obj.taskQueue, task];
                return;
            end

            % === CASO 2: Generazione random con finestra di diversità ===
            pickupNodes = find(strcmp(obj.nodeTypes, 'W_pickup'));
            dropoffNodes = find(strcmp(obj.nodeTypes, 'W_dropoff'));

            % Estrai ultimi 6 taskTypeId generati (per evitare ripetizioni in finestra di 7)
            recentTaskTypes = [];
            windowSize = min(6, length(obj.taskQueue));

            if windowSize > 0
                % Prendi gli ultimi 6 task nella coda
                recentTasks = obj.taskQueue(end-windowSize+1:end);
                recentTaskTypes = [recentTasks.taskTypeId];
            end

            % Genera tutte le combinazioni possibili
            allCombinations = [];
            for p = 1:length(pickupNodes)
                for d = 1:length(dropoffNodes)
                    taskTypeId = (p - 1) * length(dropoffNodes) + d;
                    allCombinations = [allCombinations; pickupNodes(p), dropoffNodes(d), taskTypeId];
                end
            end

            % Filtra combinazioni già usate di recente
            availableCombinations = [];
            for i = 1:size(allCombinations, 1)
                if ~ismember(allCombinations(i, 3), recentTaskTypes)
                    availableCombinations = [availableCombinations; allCombinations(i, :)];
                end
            end

            % Se tutte le combinazioni sono state usate, resetta (caso limite)
            if isempty(availableCombinations)
                availableCombinations = allCombinations;
                if obj.debugMode
                    fprintf('⚠️  Tutte le combinazioni usate negli ultimi 6 task, reset\n');
                end
            end

            % Seleziona random tra quelle disponibili
            idx = randi(size(availableCombinations, 1));
            pickup = availableCombinations(idx, 1);
            dropoff = availableCombinations(idx, 2);

            % Crea task
            task = struct();
            obj.taskCounter = obj.taskCounter + 1;
            task.sequentialId = obj.taskCounter;
            task.taskTypeId = obj.computeTaskTypeId(pickup, dropoff);
            task.pickup = pickup;
            task.dropoff = dropoff;
            task.creationTime = obj.currentTime;
            task.assignmentTime = -1;
            task.assignedAGV = 0;
            task.totalDistance = 0;

            obj.taskQueue = [obj.taskQueue, task];

            if obj.debugMode
                fprintf('✅ Task %d generato: %s → %s (typeId=%d, ultimi 6: [%s])\n', ...
                    task.sequentialId, obj.getNodeName(pickup), obj.getNodeName(dropoff), ...
                    task.taskTypeId, num2str(recentTaskTypes));
            end
        end

        function taskTypeId = computeTaskTypeId(obj, pickupNode, dropoffNode)
            pickupNodes = find(strcmp(obj.nodeTypes, 'W_pickup'));
            dropoffNodes = find(strcmp(obj.nodeTypes, 'W_dropoff'));

            pickupIndex = find(pickupNodes == pickupNode);
            dropoffIndex = find(dropoffNodes == dropoffNode);

            % ID univoco 1-14 per coppia pickup-dropoff
            taskTypeId = (pickupIndex - 1) * length(dropoffNodes) + dropoffIndex;
        end

        % === METODI PRIM (Algorithm 3, Cap. 5) ===

        function runPRIMAuction(obj)
            % Asta multi-round PRIM: assegna tutti i task nel pool alle code Qu degli AGV
            % Implementa Algorithm 3, righe 9-23

            while ~isempty(obj.primAuctionPool)
                % Riga 10: Create a set of bids B = ∅
                minBid = inf;
                winnerAGV = -1;
                winnerTask = [];
                winnerTaskIdx = -1;

                % Righe 11-19: Ogni AGV fa offerta su ogni task, si seleziona la minima
                for taskIdx = 1:length(obj.primAuctionPool)
                    task = obj.primAuctionPool(taskIdx);
                    for agvId = 1:obj.nAGV
                        bid = obj.computeBid(agvId, task);  % Righe 13-14
                        if bid < minBid
                            minBid = bid;
                            winnerAGV = agvId;
                            winnerTask = task;
                            winnerTaskIdx = taskIdx;
                        end
                    end
                end

                % Riga 21: Qu_j = Qu_j ∪ Tm (assegna task vincitore all'AGV vincitore)
                obj.AGVs(winnerAGV).unexecutedTask = [obj.AGVs(winnerAGV).unexecutedTask, winnerTask];

                % Riga 22: TA = TA \ {Tm} (rimuovi task assegnato dal pool)
                obj.primAuctionPool(winnerTaskIdx) = [];

                if obj.debugMode
                    fprintf('🎯 Asta: Task %d → AGV %d (bid=%.1f)\n', ...
                        winnerTask.sequentialId, winnerAGV, minBid);
                end
            end
        end

        function bid = computeBid(obj, agvId, task)
            % Calcola offerta per AGV su task secondo MiniSum (eq. 5.3-5.4) o MiniMax (eq. 5.5-5.6)

            % Determina punto di partenza (Def. 5.2-5.4, Cap. 5, p.119-120):
            % - Se sta eseguendo un task (Qe != ∅): parte dal dropoff del task corrente
            % - Altrimenti: parte dalla posizione corrente
            if ~isempty(obj.AGVs(agvId).task)
                lastPoint = obj.AGVs(agvId).task.dropoff;
            else
                lastPoint = obj.AGVs(agvId).logicalNode;
            end

            % Se modalità MiniMax: accumula costi di tutti i task in coda Qu
            if strcmp(obj.operatingMode, 'PRIM_MM')
                cumulativeCost = 0;
                for i = 1:length(obj.AGVs(agvId).unexecutedTask)
                    qTask = obj.AGVs(agvId).unexecutedTask(i);

                    % Costo per raggiungere pickup del task in coda
                    if lastPoint ~= qTask.pickup
                        pickupPath = obj.getOptimalPath(lastPoint, qTask.pickup);
                        cumulativeCost = cumulativeCost + obj.calculatePathDistance(pickupPath);
                    end

                    % Costo per eseguire il task in coda (pickup → dropoff)
                    taskPath = obj.getOptimalPath(qTask.pickup, qTask.dropoff);
                    cumulativeCost = cumulativeCost + obj.calculatePathDistance(taskPath);

                    lastPoint = qTask.dropoff;  % Aggiorna per prossimo task
                end
            else
                cumulativeCost = 0;  % MiniSum: ignora coda esistente
            end

            % Costo per il nuovo task auctionato
            % Costo per raggiungere pickup
            pickupDist = 0;
            if lastPoint ~= task.pickup
                pickupPath = obj.getOptimalPath(lastPoint, task.pickup);
                pickupDist = obj.calculatePathDistance(pickupPath);
            end

            % Costo per eseguire il task (pickup → dropoff)
            taskPath = obj.getOptimalPath(task.pickup, task.dropoff);
            taskDist = obj.calculatePathDistance(taskPath);

            % Bid finale dipende dalla modalità:
            % MiniSum (eq. 5.3-5.4): solo costo marginale del nuovo task
            % MiniMax (eq. 5.5-5.6): costo cumulativo di tutta la coda + nuovo task
            bid = cumulativeCost + pickupDist + taskDist;
        end

        function startNextTaskForAGV(obj, agvId)
            % Avvia esecuzione del primo task dalla coda Qu dell'AGV
            % Utilizzato da PRIM dopo asta e dopo completamento task

            if isempty(obj.AGVs(agvId).unexecutedTask)
                return;  % Nessun task in coda
            end

            % Estrae primo task dalla coda (FIFO sulla coda personale dell'AGV)
            task = obj.AGVs(agvId).unexecutedTask(1);
            obj.AGVs(agvId).unexecutedTask(1) = [];

            % Reset tempi stati per nuovo task
            obj.resetCurrentTaskStateTimes(agvId);

            % Pianifica route segmentata (corrente → pickup → dropoff)
            route = obj.planRouteForTaskSegmented(agvId, task);

            % Assegna task e route all'AGV
            obj.AGVs(agvId).task = task;
            obj.AGVs(agvId).residualRoute = route;
            obj.AGVs(agvId).task.assignmentTime = obj.currentTime;
            obj.AGVs(agvId).totalPlannedDistance = obj.calculatePathDistance(route);

            % Registra entry time se non già fatto (per calcolo delay nel pool)
            if ~obj.taskPoolEntryTimes.isKey(task.sequentialId)
                obj.taskPoolEntryTimes(task.sequentialId) = task.creationTime;
            end

            % Cambia stato a WAITING (2) per attivare traffic control
            obj.changeAGVState(agvId, obj.AGVs(agvId).state, 2);

            if obj.debugMode
                fprintf('▶️ AGV %d avvia Task %d: %s→%s (coda: %d task)\n', agvId, ...
                    task.sequentialId, obj.getNodeName(task.pickup), obj.getNodeName(task.dropoff), ...
                    length(obj.AGVs(agvId).unexecutedTask));
            end
        end

        function taskTimes = getCurrentTaskStateTimes(obj, agvId)
            % Ritorna i tempi stati per il task corrente dell'AGV
            if isempty(obj.AGVs(agvId).task)
                taskTimes = [];
                return;
            end

            % Finalizza tempo stato corrente
            currentState = obj.AGVs(agvId).state;
            currentTaskTimes = obj.AGVs(agvId).currentTaskStateTimes;

            if currentState >= 0
                stateNames = {'Idle', 'Resuming', 'Waiting', 'Resolving'};
                stateName = stateNames{currentState + 1};
                additionalTime = obj.currentTime - obj.agvStateTimes(agvId).lastTransition;
                currentTaskTimes.(stateName) = currentTaskTimes.(stateName) + additionalTime;
            end

            taskTimes = struct();
            taskTimes.Idle = currentTaskTimes.Idle;
            taskTimes.Resuming = currentTaskTimes.Resuming;
            taskTimes.Waiting = currentTaskTimes.Waiting;
            taskTimes.Resolving = currentTaskTimes.Resolving;
            taskTimes.totalTime = obj.currentTime - currentTaskTimes.taskStartTime;
        end
        
        function route = planRouteForTaskSegmented(obj, agvId, task)
            currentNode = obj.AGVs(agvId).logicalNode;

            % Calcola route e salva le chiavi dei segmenti
            if currentNode == task.pickup
                % Solo route verso dropoff
                route = obj.getOptimalPath(task.pickup, task.dropoff);
                obj.AGVs(agvId).routeKeys = {sprintf('%d_%d', task.pickup, task.dropoff)};
                obj.AGVs(agvId).finalTarget = task.dropoff;
            else
                % Route suddivisa: pickup + dropoff
                routeToPickup = obj.getOptimalPath(currentNode, task.pickup);
                routeToDropoff = obj.getOptimalPath(task.pickup, task.dropoff);
                route = [routeToPickup, routeToDropoff(2:end)];

                % Salva le chiavi dei due segmenti
                obj.AGVs(agvId).routeKeys = {
                    sprintf('%d_%d', currentNode, task.pickup),
                    sprintf('%d_%d', task.pickup, task.dropoff)
                    };
                obj.AGVs(agvId).finalTarget = task.dropoff;
            end

            % Inizializza l'indice del segmento corrente
            obj.AGVs(agvId).currentSegmentIndex = 1;
        end
        
        function route = findRouteToNearestParkingDirect(obj, agvId)
            % Route diretta al parcheggio più vicino usando route precalcolate
            currentNode = obj.AGVs(agvId).logicalNode;
            parkingNodes = find(strcmp(obj.nodeTypes, 'P'));
            
            bestParking = [];
            minDistance = inf;
            
            for p = parkingNodes'
                if obj.isParkingAvailable(p, agvId)
                    % Usa route precalcolata se disponibile
                    path = obj.getOptimalPath(currentNode, p);
                    distance = obj.calculatePathDistance(path);
                    if distance < minDistance
                        minDistance = distance;
                        bestParking = p;
                    end
                end
            end
            
            if ~isempty(bestParking)
                route = obj.getOptimalPath(currentNode, bestParking);

                % Salva informazioni per il routing verso parcheggio
                obj.AGVs(agvId).routeKeys = {sprintf('%d_%d', currentNode, bestParking)};
                obj.AGVs(agvId).currentSegmentIndex = 1;
                obj.AGVs(agvId).finalTarget = bestParking;
            else
                route = [];
            end
        end
        
        function [Nx, Ny] = getTravellingInformation(obj, agvId)
            agv = obj.AGVs(agvId);
            Nx = agv.logicalNode;
            
            if ~isempty(agv.residualRoute) && length(agv.residualRoute) > 1
                Ny = agv.residualRoute(2);
            else
                Ny = [];
            end
        end
        
        function updateResidualRouteFromTI(obj, agvId, Nx, Ny)
            agv = obj.AGVs(agvId);
            
            if ~isempty(agv.residualRoute) && agv.residualRoute(1) ~= Nx
                if obj.debugMode
                    warning('Inconsistenza TI per AGV %d: route[1]=%d, Nx=%d', ...
                        agvId, agv.residualRoute(1), Nx);
                end
            end
        end
        
        function createSharedRoute(obj, agvId1, agvId2)
            Li = obj.AGVs(agvId1).residualRoute;
            Lj = obj.AGVs(agvId2).residualRoute;
            
            if isempty(Li) || isempty(Lj)
                sharedRoute = [];
            else
                sharedRoute = intersect(Li, Lj, 'stable');
            end
            
            key = sprintf('%d_%d', agvId1, agvId2);
            obj.sharedRoutes(key) = sharedRoute;
        end
        
        function updateWholeSharedRoute(obj, agvId)
            wholeSharedRoute = [];
            
            for otherAgvId = 1:obj.nAGV
                if otherAgvId ~= agvId
                    key = sprintf('%d_%d', agvId, otherAgvId);
                    if obj.sharedRoutes.isKey(key)
                        pairSharedRoute = obj.sharedRoutes(key);
                        wholeSharedRoute = union(wholeSharedRoute, pairSharedRoute, 'stable');
                    end
                end
            end
            
            obj.AGVs(agvId).sharedRoute = wholeSharedRoute;
        end
        
        function executeTrafficRules(obj, agvId)
            obj.applyRule1(agvId);
        end
        
        % === IMPLEMENTAZIONE DELLE 10 REGOLE IDRR ===
        function applyRule1(obj, agvId)
            % Rule 1: Check pending maneuvers o active maneuvers
            if obj.hasPendingManeuver(agvId)
                if ~obj.canExecuteManeuver(agvId)
                    obj.setAGVWaiting(agvId);
                    return;
                end
                obj.executeManeuver(agvId);
                return;
            elseif obj.hasActiveManeuver(agvId)
                obj.changeAGVState(agvId, obj.AGVs(agvId).state, 3); % resolving
                return;
            else
                obj.applyRule2(agvId);
            end
        end
        
        function applyRule2(obj, agvId)
            % Rule 2: Check SNx = 2
            currentNode = obj.AGVs(agvId).logicalNode;
            if obj.nodeStates(currentNode) == 2
                obj.applyRule3(agvId);
            else
                obj.applyRule5(agvId);
            end
        end
        
        function applyRule3(obj, agvId)
            % Rule 3: Check if other AGV at exit CP
            [hasConflict, otherAGV] = obj.checkExitCPBlocked(agvId);
            if hasConflict
                obj.applyRule4(agvId, otherAGV);
            else
                obj.applyRule5(agvId);
            end
        end
        

        function applyRule4(obj, agvId, otherAGV)
            % Rule 4 (paper): check Sj=3; else request Res2.
            % Note: per paper there is NO additional exit-CP guard here.
            % The Rule 5/7 continuation happens inside completeManeuver once
            % Res2 actually finishes (requestRes2 always returns success=false).
            if obj.AGVs(otherAGV).state == 3
                obj.setAGVWaiting(agvId);
                return;
            end

            if obj.hasPendingManeuver(otherAGV) || obj.hasActiveManeuver(otherAGV)
                obj.setAGVWaiting(agvId);
                return;
            end

            % Request Res2. Always returns success=false (Ri must wait for
            % maneuver completion). Rule 5/7 continuation is in completeManeuver.
            obj.requestRes2(agvId, otherAGV);
        end

        function ok = res2CanFreeExit(obj, requestingAGV, blockingAGV)
            % La Res2 e' utile solo se il bloccante puo' essere spostato su un
            % exit CP diverso da quello che ostruisce il richiedente.
            ok = false;
            nodeIdx = obj.AGVs(requestingAGV).logicalNode;

            nextReq = obj.getNextNodeInRoute(requestingAGV);
            nextBlk = obj.getNextNodeInRoute(blockingAGV);
            if isempty(nextReq) || isempty(nextBlk), return; end

            [~, exitReq] = obj.getCPsFromPrecomputedTable(requestingAGV, nextReq);
            [~, exitBlk] = obj.getCPsFromPrecomputedTable(blockingAGV, nextBlk);

            % Utile solo se gli exit CP sono distinti (no pursuit/stessa uscita)
            ok = (exitReq ~= exitBlk);
        end

        function applyRule5(obj, agvId)
            % Rule 5: Check SNy = 2
            nextNode = obj.getNextNodeInRoute(agvId);  
            if obj.nodeStates(nextNode) == 2
                obj.setAGVWaiting(agvId);
                return; % EXIT
            else
                obj.applyRule6(agvId);
            end
        end
        
        function applyRule6(obj, agvId)
            % Rule 6: Check cycle pursuit-free condition
            nextNode = obj.getNextNodeInRoute(agvId);
            if ~obj.checkCyclePursuitFree(nextNode, agvId)
                obj.setAGVWaiting(agvId);
                return; % EXIT
            else
                obj.applyRule7(agvId);
            end
        end
        
        function applyRule7(obj, agvId)
            % Rule 7: Check SNy state and type
            nextNode = obj.getNextNodeInRoute(agvId);
            nodeOccupancy = obj.nodeStates(nextNode);
            nodeType = obj.nodes(nextNode).type;
            
            if nodeOccupancy == 0
                obj.setAGVToResuming(agvId);
                return; % EXIT
            elseif nodeOccupancy == 1 && (strcmp(nodeType, 'P') || contains(nodeType, 'W'))
                obj.setAGVWaiting(agvId);
                return; % EXIT
            else % nodeOccupancy == 1 AND nextNode ∈ TX
                obj.applyRule8(agvId);
            end
        end
        
        function applyRule8(obj, agvId)

            [conflictAGV, conflictType] = obj.findConflictInNextNode(agvId);
            Nx = obj.AGVs(agvId).logicalNode;
            isTX = strcmp(obj.nodes(Nx).type, 'T') || strcmp(obj.nodes(Nx).type, 'X');
            isWP = strcmp(obj.nodes(Nx).type, 'P') || contains(obj.nodes(Nx).type, 'W');

            if strcmp(conflictType, 'intersection')
                % Intersection: Rule 9 necessaria (nessuna Res1, Ri entra direttamente)
                obj.applyRule9(agvId, conflictAGV);

            elseif strcmp(conflictType, 'head-on')
                % Head-on: Res1 gestisce la saturazione temporanea internamente.
                % Rule 9 è sovra-conservativa qui: salta direttamente a Rule 10.
                obj.applyRule10(agvId, conflictAGV);

            else
                % Pursuit: Ri aspetta distanza di sicurezza
                obj.setAGVWaiting(agvId);
            end
        end
        
        function applyRule9(obj, agvId, conflictAGV)
            % Rule 9: Check path saturation
            if obj.checkPathSaturation(agvId, conflictAGV)
                obj.setAGVWaiting(agvId);
                return; % EXIT
            else
                obj.applyRule10(agvId, conflictAGV);
            end
        end

        function applyRule10(obj, agvId, conflictAGV)
            % Rule 10: Check Rk position and decide Res1 or resume.
            % Prima di qualsiasi movimento verso Ny, verifica che la mossa di Ri
            % non crei due TX consecutivi saturi nella sua residual route.
            % (Verma et al. 2024 - nota Fig.9: adjacent TX-nodes in residual
            % routes are never saturated simultaneously)

            Nx     = obj.AGVs(agvId).logicalNode;
            Ny     = obj.getNextNodeInRoute(agvId);
            % route1 = obj.AGVs(agvId).residualRoute;
            % 
            % % Simula stato dopo Ri lascia Nx e arriva in Ny
            % sAfter       = obj.nodeStates;
            % sAfter(Nx)   = sAfter(Nx) - 1;
            % sAfter(Ny)   = sAfter(Ny) + 1;
            % 
            % % Cerca coppie consecutive di TX entrambe sature nella residual route
            % for k = 1 : numel(route1) - 1
            %     n1 = route1(k);
            %     n2 = route1(k+1);
            %     isT1 = strcmp(obj.nodes(n1).type, 'T') || strcmp(obj.nodes(n1).type, 'X');
            %     isT2 = strcmp(obj.nodes(n2).type, 'T') || strcmp(obj.nodes(n2).type, 'X');
            %     if isT1 && isT2 && sAfter(n1) >= 2 && sAfter(n2) >= 2
            %         obj.setAGVWaiting(agvId);
            %         return;
            %     end
            % end

            % Check superato: procedi con Rule 10 standard
            [entryCP, ~] = obj.getCPsFromPrecomputedTable(agvId, Ny);

            if ~obj.isAGVAtCP(conflictAGV, Ny, entryCP)
                obj.setAGVToResuming(agvId);
                return;
            else
                if obj.hasPendingManeuver(conflictAGV) || obj.hasActiveManeuver(conflictAGV)
                    obj.setAGVWaiting(agvId);
                    if obj.debugMode
                        fprintf('⏸️  AGV %d: conflictAGV %d già in manovra, waiting\n', agvId, conflictAGV);
                    end
                    return;
                end
                success = obj.requestRes1(agvId, conflictAGV, Ny);
                if ~success
                    return;
                end
                return;
            end
        end
        
        % === FUNZIONI DI SUPPORTO PER LE REGOLE ===
        function nextNode = getNextNodeInRoute(obj, agvId)
            residualRoute = obj.AGVs(agvId).residualRoute;
            if length(residualRoute) > 1
                nextNode = residualRoute(2);
            else
                nextNode = [];
            end
        end
        
        function success = setAGVToResuming(obj, agvId)
            agv = obj.AGVs(agvId);
            currentNode = agv.logicalNode;
            currentCP = agv.logicalCP;
            nextNode = obj.getNextNodeInRoute(agvId);
            
            if isempty(nextNode)
                success = false;
                return;
            end
            
            [entryCP, ~] = obj.getCPsFromPrecomputedTable(agvId, nextNode);
            
            % Operazione atomica
            success = obj.atomicStateTransition(agvId, currentNode, currentCP, nextNode, entryCP);
                      
            if success
                obj.AGVs(agvId).logicalNode = nextNode;
                obj.AGVs(agvId).logicalCP = entryCP;
                obj.AGVs(agvId).isMoving = true;
                obj.changeAGVState(agvId, obj.AGVs(agvId).state, 1); % resuming
                
                distance = obj.edges(currentNode, nextNode);
                obj.AGVs(agvId).arrivalTime = obj.currentTime + distance / obj.agvSpeed;
                obj.AGVs(agvId).totalDistance = obj.AGVs(agvId).totalDistance + distance;
                if ~isempty(obj.AGVs(agvId).task)
                    obj.AGVs(agvId).taskDistance = obj.AGVs(agvId).taskDistance + distance;
                end
                if obj.debugMode
                    currentNodeName = obj.getNodeName(currentNode);
                    nextNodeName = obj.getNodeName(nextNode);
                    fprintf('🔄 AGV %d: prenotato %s(CP%d) → %s(CP%d)\n', agvId, ...
                        currentNodeName, currentCP, nextNodeName, entryCP);
                end
            else
                obj.setAGVWaiting(agvId);
            end
        end
        
        function setAGVWaiting(obj, agvId)
            obj.changeAGVState(agvId, obj.AGVs(agvId).state, 2);
            obj.AGVs(agvId).waitingTime = obj.AGVs(agvId).waitingTime + obj.timeStep;
        end
        
        function success = atomicStateTransition(obj, agvId, oldNode, oldCP, newNode, newCP)
            success = false;
            
            newCP_globalIdx = obj.getGlobalCPIndex(newNode, newCP);
            oldCP_globalIdx = obj.getGlobalCPIndex(oldNode, oldCP);
            
            if obj.cpStates(newCP_globalIdx) ~= 0
                return;
            end
            
            if obj.cpOccupants(oldCP_globalIdx) ~= agvId
                error('AGV %d non occupa il CP specificato', agvId);
            end
            
            try
                % Rilascia CP corrente
                obj.cpStates(oldCP_globalIdx) = 0;
                obj.cpOccupants(oldCP_globalIdx) = 0;
                
                % Prenota CP destinazione
                obj.cpStates(newCP_globalIdx) = 1;
                obj.cpOccupants(newCP_globalIdx) = agvId;
                
                % Aggiorna stati nodi
                obj.nodeStates(oldNode) = obj.nodeStates(oldNode) - 1;
                obj.nodeStates(newNode) = obj.nodeStates(newNode) + 1;
                
                % Aggiorna liste occupanti
                obj.nodeOccupants{oldNode}(obj.nodeOccupants{oldNode} == agvId) = [];
                obj.nodeOccupants{newNode} = [obj.nodeOccupants{newNode}, agvId];
                
                success = true;
                
            catch ME
                warning('Errore transizione AGV %d: %s', agvId, ME.message);
                success = false;
            end
        end
        
        function globalIdx = getGlobalCPIndex(obj, nodeIdx, localCP)
            cpData = obj.controlPoints(nodeIdx);
            globalIdx = cpData.globalIndices(localCP);
        end
        
        % === GESTIONE MOVIMENTI FISICI ===
        function updatePhysicalMovements(obj)
            for agvId = 1:obj.nAGV
                agv = obj.AGVs(agvId);

                if agv.isMoving && obj.currentTime >= agv.arrivalTime

                    % Controlla se questo è un movimento intra-nodo Res1
                    % (blockingAGV che si sposta da entryCP a targetCP)
                    res1Idx = obj.findActiveRes1ForBlocker(agvId);

                    if ~isempty(res1Idx)
                        % === Arrivo intra-nodo: completamento Res1 ===
                        obj.completeRes1Movement(agvId, res1Idx);
                    else
                        % === Arrivo inter-nodo: logica standard ===
                        obj.AGVs(agvId).isMoving = false;
                        obj.changeAGVState(agvId, obj.AGVs(agvId).state, 2); % waiting

                        if obj.debugMode
                            nodeName = obj.getNodeName(agv.logicalNode);
                            cpInfo = sprintf('CP%d', agv.logicalCP);
                            fprintf('🎯 AGV %d arrivato fisicamente in %s(%s)\n', agvId, nodeName, cpInfo);
                        end

                        obj.updateResidualRoute(agvId);
                        obj.checkPendingManeuversForAGV(agvId);
                    end
                end
            end
            % Promotion pass post-arrivo: un AGV appena arrivato
            % (isMoving: true→false in questo stesso timestep) potrebbe
            % aver reso pronta una Res2 pending che updateActiveManeuvers
            % non ha potuto promuovere (girava prima degli arrivi).
            for i = length(obj.pendingManeuvers):-1:1
                m = obj.pendingManeuvers(i);
                if obj.AGVs(m.agv1).state == 2 && obj.AGVs(m.agv2).state == 2 && ...
                        ~obj.AGVs(m.agv1).isMoving && ~obj.AGVs(m.agv2).isMoving
                    m.startTime = obj.currentTime;
                    m.endTime   = obj.currentTime + obj.res2Duration;
                    obj.AGVs(m.agv1).state = 3;
                    obj.AGVs(m.agv2).state = 3;
                    obj.activeManeuvers  = [obj.activeManeuvers, m];
                    obj.pendingManeuvers(i) = [];
                    if obj.debugMode
                        fprintf('⚡ [postArrival] Res2 promossa: AGV%d-AGV%d in %s\n', ...
                            m.agv1, m.agv2, obj.getNodeName(m.nodeIdx));
                    end
                end
            end
        end

        function idx = findActiveRes1ForBlocker(obj, agvId)
            % Restituisce l'indice in activeManeuvers della Res1 dove agvId è il
            % blockingAGV (agv2). Restituisce [] se non esiste.
            idx = [];
            for i = 1:length(obj.activeManeuvers)
                m = obj.activeManeuvers(i);
                if strcmp(m.type, 'Res1') && m.agv2 == agvId
                    idx = i;
                    return;
                end
            end
        end

        function completeRes1Movement(obj, blockerAGV, maneuverIdx)
            % Chiamata da updatePhysicalMovements quando blockingAGV arriva a targetCP.
            % Libera entryCP, aggiorna logicalCP di Rk, avvia requestingAGV verso Ny.

            maneuver      = obj.activeManeuvers(maneuverIdx);
            requestingAGV = maneuver.agv1;

            % Rk ha completato il movimento intra-nodo
            obj.AGVs(blockerAGV).isMoving = false;

            % Swap CP: libera entryCP, imposta logicalCP=targetCP per Rk, Rk.state=2
            obj.executeRes1(maneuver);

            if obj.debugMode
                fprintf('✅ Res1 completata: AGV%d arrivato a CP%d in %s\n', ...
                    blockerAGV, maneuver.targetCP, obj.getNodeName(maneuver.nodeIdx));
            end

            % requestingAGV riprende verso Ny (entryCP ora libero)
            success = obj.setAGVToResuming(requestingAGV);
            if ~success
                obj.setAGVWaiting(requestingAGV);
                if obj.debugMode
                    fprintf('⚠️  completeRes1Movement: setAGVToResuming fallita per AGV%d\n', requestingAGV);
                end
            end

            % Rimuovi da activeManeuvers
            obj.activeManeuvers(maneuverIdx) = [];
            obj.applyPostManeuverRules(blockerAGV);
        end

        function checkPendingManeuversForAGV(obj, agvId)
            % Quando un AGV arriva, resta in waiting
            % La Rule 1 controllerà le pending maneuvers al prossimo processamento

            if obj.debugMode
                for i = 1:length(obj.pendingManeuvers)
                    maneuver = obj.pendingManeuvers(i);
                    if (maneuver.agv1 == agvId || maneuver.agv2 == agvId)
                        fprintf('🔧 AGV %d arrivato, manovra %s sarà gestita dalla Rule 1\n', ...
                            agvId, maneuver.type);
                        break;
                    end
                end
            end
        end
        
        function updateResidualRoute(obj, agvId)
            agv = obj.AGVs(agvId);
            if ~isempty(agv.residualRoute) && length(agv.residualRoute) > 1
                obj.AGVs(agvId).residualRoute(1) = [];

                % Controlla se abbiamo completato un segmento
                obj.updateSegmentIndex(agvId);
            elseif length(agv.residualRoute) == 1
                obj.AGVs(agvId).residualRoute = [];
                obj.AGVs(agvId).currentSegmentIndex = 0;
                obj.AGVs(agvId).routeKeys = {};
            end
        end

        % function updateSegmentIndex(obj, agvId)
        %     % Aggiorna l'indice del segmento quando necessario
        % 
        %     agv = obj.AGVs(agvId);
        % 
        %     if ~isempty(agv.task) && ~isempty(agv.routeKeys) && length(agv.routeKeys) > 1
        %         % Se abbiamo un task con pickup e siamo arrivati al pickup
        %         if agv.logicalNode == agv.task.pickup && agv.currentSegmentIndex == 1
        %             obj.AGVs(agvId).currentSegmentIndex = 2;
        %             if obj.debugMode
        %                 fprintf('AGV %d: passaggio al segmento 2 (pickup→dropoff)\n', agvId);
        %             end
        %         end
        %     end
        % end

        function updateSegmentIndex(obj, agvId)
            agv = obj.AGVs(agvId);
            if ~isempty(agv.task) && ~isempty(agv.routeKeys) && length(agv.routeKeys) > 1
                if agv.logicalNode == agv.task.pickup && agv.currentSegmentIndex == 1
                    obj.AGVs(agvId).currentSegmentIndex = 2;
                    % ← NUOVO: AGV parte dalla pickup → genera task reattivo
                    if obj.eventDrivenMode
                        obj.injectEventDrivenTask(agv.task.pickup);
                    end
                    if obj.debugMode
                        fprintf('AGV %d: passaggio al segmento 2 (pickup→dropoff)\n', agvId);
                    end
                end
            end
        end

        function handleTaskCompletion(obj, agvId)
            agv = obj.AGVs(agvId);
            currentNodeType = obj.nodes(agv.logicalNode).type;

            % CASO 1: Completamento task regolare (pickup/dropoff)
            if ~isempty(agv.task) && agv.logicalNode == agv.task.dropoff
                completedTask = agv.task;
                completedTask.completionTime = obj.currentTime;
                completedTask.assignedAGV = agvId;
                completedTask.totalDistance = agv.taskDistance;

                % Calcola delay nel pool (da ingresso pool a assegnazione)
                taskId = completedTask.sequentialId;
                if obj.taskPoolEntryTimes.isKey(taskId)
                    poolEntryTime = obj.taskPoolEntryTimes(taskId);
                    delayInPool = completedTask.assignmentTime - poolEntryTime;
                    obj.taskDelaysInPool = [obj.taskDelaysInPool, delayInPool];

                    % Rimuovi dalla mappa (pulizia memoria)
                    obj.taskPoolEntryTimes.remove(taskId);
                end

                obj.completedTasks = [obj.completedTasks, completedTask];
                
                % Incrementa contatore pickup station
                pickupNode = completedTask.pickup;
                if obj.pickupStationCounts.isKey(pickupNode)
                    obj.pickupStationCounts(pickupNode) = obj.pickupStationCounts(pickupNode) + 1;
                end

                % AGGIUNTA RL: Update throughput tracking
                obj.updateThroughputOnTaskCompletion(agvId);

                if strcmp(obj.operatingMode, 'RL_TRAINING') || strcmp(obj.operatingMode, 'RL_TESTING')
                    obj.notifyTaskCompletion(agvId, completedTask);
                end

                obj.AGVs(agvId).task = [];
                obj.AGVs(agvId).tasksCompleted = obj.AGVs(agvId).tasksCompleted + 1;

                % Svuota completamente i campi di routing
                obj.AGVs(agvId).residualRoute = [];
                obj.AGVs(agvId).routeKeys = {};
                obj.AGVs(agvId).currentSegmentIndex = 0;
                obj.AGVs(agvId).finalTarget = [];
                obj.AGVs(agvId).sharedRoute = [];
                obj.AGVs(agvId).taskDistance = 0;

                % PRIM: Controlla coda Qu prima di andare idle
                if strcmp(obj.operatingMode, 'PRIM_MS') || strcmp(obj.operatingMode, 'PRIM_MM')
                    if ~isempty(obj.AGVs(agvId).unexecutedTask)
                        % Ha altri task in coda → avvia il prossimo immediatamente
                        obj.startNextTaskForAGV(agvId);
                    else
                        % Nessun task in coda → idle + vai al parcheggio se in workstation
                        obj.changeAGVState(agvId, obj.AGVs(agvId).state, 0);

                        currentNodeType = obj.nodes(obj.AGVs(agvId).logicalNode).type;
                        if contains(currentNodeType, 'W')
                            route = obj.findRouteToNearestParkingDirect(agvId);
                            if ~isempty(route)
                                obj.AGVs(agvId).residualRoute = route;
                                obj.AGVs(agvId).totalPlannedDistance = obj.calculatePathDistance(route);
                                obj.changeAGVState(agvId, 0, 2);  % idle → waiting
                                if obj.debugMode
                                    fprintf('🅿️ AGV %d completato Task %d → parcheggio %s\n', ...
                                        agvId, completedTask.sequentialId, obj.getNodeName(route(end)));
                                end
                            end
                        end
                    end
                else
                    % FIFO/RL: comportamento originale (transizione immediata a idle)
                    obj.changeAGVState(agvId, obj.AGVs(agvId).state, 0);

                    if obj.debugMode
                        dropoffName = obj.getNodeName(completedTask.dropoff);
                        pickupName = obj.getNodeName(completedTask.pickup);
                        fprintf('✅ AGV %d completato Task %d (%s → %s) → IDLE\n', ...
                            agvId, completedTask.sequentialId, pickupName, dropoffName);
                    end
                end

                % CASO 2: Arrivo al parcheggio senza task (movimento di repositioning)
            elseif isempty(agv.task) && strcmp(currentNodeType, 'P')
                % AGV è arrivato al parcheggio dopo essere stato mandato lì dal sistema

                % Svuota completamente i campi di routing
                obj.AGVs(agvId).residualRoute = [];
                obj.AGVs(agvId).routeKeys = {};
                obj.AGVs(agvId).currentSegmentIndex = 0;
                obj.AGVs(agvId).finalTarget = [];
                obj.AGVs(agvId).sharedRoute = [];
                obj.AGVs(agvId).taskDistance = 0;


                % Transizione immediata a idle
                obj.changeAGVState(agvId, obj.AGVs(agvId).state, 0); % idle

                if obj.debugMode
                    parkingName = obj.getNodeName(agv.logicalNode);
                    fprintf('🅿️  AGV %d arrivato al parcheggio %s → IDLE\n', agvId, parkingName);
                end
            end
        end

        function notifyTaskCompletion(obj, agvId, completedTask)
            % Notifica Environment per calcolo reward alla fine del task

            if isempty(obj.environmentInterface)
                if obj.verboseLogging
                    obj.logMessage('⚠️ Environment non connesso per reward calculation');
                end
                return;
            end

            if ~isempty(obj.environmentInterface) && ...
                    ismember(obj.operatingMode, {'RL_TRAINING', 'RL_TESTING'})

                % ⭐ AGGIUNTA: Controlla se episodio completato per questo AGV
                isDone = obj.checkIfEpisodeComplete(agvId);

                % Passa il flag done all'environment
                obj.environmentInterface.handleTaskCompletion(agvId, completedTask, isDone);

                if obj.verboseLogging && isDone
                    fprintf('🎯 AGV_%d: Episodio completato (done=1)\n', agvId);
                end
            end
        end

        function isDone = checkIfEpisodeComplete(obj, agvId)
            % Controlla se episodio è terminato per questo AGV specifico

            % Condizioni per done=1:
            % 1. Tutti i 21 task dell'episodio generati
            allTasksGenerated = (obj.tasksGenerated >= obj.maxTasks);

            % 2. Coda globale vuota
            globalQueueEmpty = isempty(obj.taskQueue);


            isDone = allTasksGenerated && globalQueueEmpty;

    
            if obj.verboseLogging && isDone
                fprintf('   ✅ Episode completed for AGV_%d: tasks=%d/21', ...
                    agvId, obj.tasksGenerated, globalQueueEmpty);
            end
        end
        
        % === FUNZIONI AUSILIARIE ===
              
        function path = getOptimalPath(obj, fromNode, toNode)
            key = sprintf('%d_%d', fromNode, toNode);
            if obj.symbolicRoutingTable.isKey(key)
                path = obj.symbolicRoutingTable(key);
            else
                path = shortestpath(obj.graph, fromNode, toNode);
            end
        end
        
        function distance = calculatePathDistance(obj, path)
            distance = 0;
            for i = 1:length(path)-1
                distance = distance + obj.edges(path(i), path(i+1));
            end
        end
        
        function available = isParkingAvailable(obj, parkingNode, excludeAGV)
            currentOccupancy = obj.nodeStates(parkingNode);
            capacity = obj.nodes(parkingNode).capacity;

            % Conta AGV che hanno questo parcheggio come destinazione finale
            agvsHeadingToParking = 0;

            for agvId = 1:obj.nAGV
                if agvId == excludeAGV
                    continue;
                end

                agv = obj.AGVs(agvId);

                % Se l'AGV ha una residual route e la destinazione finale è questo parcheggio
                if ~isempty(agv.residualRoute) && length(agv.residualRoute) > 0
                    finalDestination = agv.residualRoute(end);
                    if finalDestination == parkingNode
                        agvsHeadingToParking = agvsHeadingToParking + 1;
                    end
                end
            end

            totalFutureOccupancy = currentOccupancy + agvsHeadingToParking;
            available = (totalFutureOccupancy < capacity);
        end
        
        % === 1. CONFLICT DETECTION E CLASSIFICATION (Definition 7) ===
        function [conflictAGV, conflictType] = findConflictInNextNode(obj, agvId)
            
            % Prima verifica se esiste shared route
            if isempty(obj.AGVs(agvId).sharedRoute)
                conflictAGV = [];
                conflictType = 'none';
                return;
            end
            
            % Trova conflitti nel prossimo nodo secondo Definition 7
            conflictAGV = [];
            conflictType = 'none';
            
            nextNode = obj.getNextNodeInRoute(agvId);
            if isempty(nextNode)
                return;
            end
            
            % AGV presenti nel prossimo nodo
            agvsInNextNode = obj.nodeOccupants{nextNode};
            
            for otherAGV = agvsInNextNode
                if otherAGV ~= agvId
                    conflictType = obj.classifyConflictBetweenAGVs(agvId, otherAGV);
                    if ~strcmp(conflictType, 'none')
                        conflictAGV = otherAGV;
                        return;
                    end
                end
            end
        end

        function conflictType = classifyConflictBetweenAGVs(obj, agvId1, agvId2)
            % Implementa Definition 7: classificazione conflitti
            Li = obj.AGVs(agvId1).residualRoute;
            Lj = obj.AGVs(agvId2).residualRoute;

            if isempty(Li) || isempty(Lj) || length(Li) < 2 || length(Lj) < 2
                conflictType = 'none';
                return;
            end

            % Trova posizioni correnti nelle residual routes
            currentNode1 = obj.AGVs(agvId1).logicalNode;
            currentNode2 = obj.AGVs(agvId2).logicalNode;

            ki = find(Li == currentNode1, 1);
            kj = find(Lj == currentNode2, 1);

            if isempty(ki) || isempty(kj) || ki >= length(Li) || kj >= length(Lj)
                conflictType = 'none';
                return;
            end

            % Nodi secondo Definition 7
            Tiki = Li(ki);
            Tjkj = Lj(kj);
            Tiki_plus1 = Li(ki + 1);
            Tjkj_plus1 = Lj(kj + 1);

            % CREA CHIAVE CONFLITTO (come nella versione precedente)
            conflictKey = sprintf('%d_%d_%d_%d_%d_%d', agvId1, agvId2, Tiki, Tiki_plus1, Tjkj, Tjkj_plus1);

            % Definition 7.i: Head-on conflict
            if Tiki == Tjkj_plus1 && Tiki_plus1 == Tjkj
                conflictType = 'head-on';
                % CONTA SOLO SE NON GIÀ PRESENTE
                if ~obj.activeConflicts.isKey(conflictKey)
                    obj.metrics.conflictCounts.headon = obj.metrics.conflictCounts.headon + 1;
                    obj.activeConflicts(conflictKey) = true;
                end

                % Definition 7.ii: Intersection conflict
            elseif Tiki_plus1 == Tjkj && (ki + 2 > length(Li) || Li(ki + 2) ~= Tjkj_plus1)
                conflictType = 'intersection';
                if ~obj.activeConflicts.isKey(conflictKey)
                    obj.metrics.conflictCounts.intersection = obj.metrics.conflictCounts.intersection + 1;
                    obj.activeConflicts(conflictKey) = true;
                end

                % Definition 7.iii: Pursuit conflict
            elseif Tiki_plus1 == Tjkj && ki + 2 <= length(Li) && Li(ki + 2) == Tjkj_plus1
                conflictType = 'pursuit';
                if ~obj.activeConflicts.isKey(conflictKey)
                    obj.metrics.conflictCounts.pursuit = obj.metrics.conflictCounts.pursuit + 1;
                    obj.activeConflicts(conflictKey) = true;
                end

            else
                conflictType = 'none';
            end
        end
        
        % === 2. CYCLE PURSUIT-FREE CONDITION (Definitions 13, 14) ===
        function pursuitFree = checkCyclePursuitFree(obj, nodeIdx, agvId)
            % Implementa Definition 14: cycle pursuit-free condition
            % Verifica se allocare agvId in nodeIdx crea un loop conflict

            pursuitFree = true;

            % Solo nodi T/X possono appartenere a cicli
            nodeType = obj.nodes(nodeIdx).type;
            if ~(strcmp(nodeType, 'T') || strcmp(nodeType, 'X'))
                return;
            end

            if ~obj.nodeCycleMap.isKey(nodeIdx)
                return; % Nodo non appartiene a nessun ciclo
            end

            cycleIndices = obj.nodeCycleMap(nodeIdx);

            % Verifica ogni ciclo secondo Definition 14
            for cycleIdx = cycleIndices
                cycle = obj.allCycles{cycleIdx};
                if obj.isCycleTroubledWithAGVAllocation(cycle, nodeIdx, agvId)
                    pursuitFree = false;

                    if obj.debugMode
                        fprintf('Loop conflict rilevato nel ciclo: [%s] con AGV %d → %s\n', ...
                            strjoin(arrayfun(@(x) obj.getNodeName(x), cycle, 'UniformOutput', false), ', '), ...
                            agvId, obj.getNodeName(nodeIdx));
                    end

                    obj.metrics.conflictCounts.loop = obj.metrics.conflictCounts.loop + 1;
                    return;
                end
            end
        end
               
        function troubled = isCycleTroubledWithAGVAllocation(obj, cycle, targetNodeIdx, movingAGVId)
            % Definition 13: verifica se ALLOCARE movingAGVId nel nodo targetNodeIdx (Ny)
            % crea un loop conflict nel ciclo "cycle".
            %
            % Coerente con Def. 13-14: si simula SEMPRE l'allocazione in Ny, sia che
            % l'AGV provenga dall'esterno del ciclo sia dal suo interno. La mappa e'
            % per-AGV: spostando solo movingAGVId, la posizione di partenza resta
            % occupata se vi e' un altro AGV, e diventa libera solo se l'AGV se n'e'
            % davvero andato -> nessun gap artificiale.

            troubled = false;

            % Raccogli gli AGV attualmente presenti nei nodi del ciclo
            agvsInCycle = [];
            agvPositionMap = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
            for nodeIdx = cycle
                for agvId = obj.nodeOccupants{nodeIdx}
                    agvsInCycle = [agvsInCycle, agvId];
                    agvPositionMap(int32(agvId)) = int32(nodeIdx);
                end
            end

            % Simula l'allocazione di movingAGVId in targetNodeIdx (Ny)
            agvPositionMap(int32(movingAGVId)) = int32(targetNodeIdx);

            % Assicura che movingAGVId sia in agvsInCycle e in testa, cosi'
            % findConsecutiveAGVsInCycle lo elegga come rappresentante del suo nodo
            agvsInCycle = [movingAGVId, agvsInCycle(agvsInCycle ~= movingAGVId)];

            troubled = obj.hasLoopConflictInCycle(cycle, agvsInCycle, ...
                agvPositionMap, movingAGVId, targetNodeIdx);
        end

        function hasLoopConflict = hasLoopConflictInCycle(obj, cycle, agvsInCycle, agvPositionMap, movingAGVId, targetNodeIdx)
            % Definition 7.iv: verifica se esiste un loop conflict
            % "Let Ri, Rj, . . . , Rk, Rl be such that their positions Pi, Pj, . . . , Pk, Pl
            % are adjacent and form a cycle in the corresponding graph, G; if every pair
            % (Ri, Rj), . . . , (Rk, Rl), (Rl, Ri) has a pursuit conflict as described in
            % item (iii), then Ri, Rj, . . . , Rk, Rl have a loop conflict"

            hasLoopConflict = false;

            if length(agvsInCycle) < 3
                return; % Serve almeno 3 AGV per un loop conflict
            end

            % Cerca sequenze di AGV in posizioni consecutive del ciclo
            for startIdx = 1:length(cycle)
                agvSequence = obj.findConsecutiveAGVsInCycle(cycle, startIdx, agvsInCycle, agvPositionMap);

                if length(agvSequence) >= 3
                    % Verifica se questa sequenza forma un loop conflict
                    if obj.isClosedPursuitChain(agvSequence, agvPositionMap, movingAGVId, targetNodeIdx)
                        hasLoopConflict = true;
                        return;
                    end
                end
            end
        end

        function agvSequence = findConsecutiveAGVsInCycle(obj, cycle, startIdx, agvsInCycle, agvPositionMap)
            % Trova una sequenza di AGV in posizioni consecutive nel ciclo

            agvSequence = [];
            cycleLength = length(cycle);

            for i = 0:cycleLength-1
                nodeIdx = cycle(mod(startIdx + i - 1, cycleLength) + 1);

                % Trova AGV in questo nodo
                agvInNode = [];
                for agvId = agvsInCycle
                    if agvPositionMap(agvId) == nodeIdx
                        agvInNode = agvId;
                        break; % Prendi il primo AGV trovato
                    end
                end

                if ~isempty(agvInNode)
                    agvSequence = [agvSequence, agvInNode];
                else
                    % Interruzione nella sequenza consecutiva
                    if length(agvSequence) >= 3
                        break; % Mantieni la sequenza trovata
                    else
                        agvSequence = []; % Reset se sequenza troppo corta
                    end
                end
            end
        end

        function isChain = isClosedPursuitChain(obj, agvSequence, agvPositionMap, movingAGVId, targetNodeIdx)
            % Definition 7.iv: verifica catena chiusa di pursuit conflicts
            % Considera la simulazione dell'AGV movingAGVId nel targetNodeIdx

            isChain = false;
            seqLength = length(agvSequence);

            if seqLength < 3
                return;
            end

            pursuitCount = 0;

            % Verifica pursuit conflicts tra AGV consecutivi nella sequenza
            for i = 1:seqLength
                agv1 = agvSequence(i);
                agv2 = agvSequence(mod(i, seqLength) + 1); % AGV successivo (circolare)

                conflictType = obj.classifyConflictWithSimulation(agv1, agv2, agvPositionMap, movingAGVId, targetNodeIdx);
                if strcmp(conflictType, 'pursuit')
                    pursuitCount = pursuitCount + 1;
                end
            end

            % È una catena chiusa se TUTTI i link sono pursuit conflicts
            isChain = (pursuitCount == seqLength);
        end
        
        
        % === NUOVI METODI PER CYCLE DETECTION ===

        function conflictType = classifyConflictWithSimulation(obj, agvId1, agvId2, agvPositionMap, movingAGVId, targetNodeIdx)
            % Classifica conflitto considerando la simulazione dell'allocazione

            % Ottieni residual routes, modificandole se necessario per la simulazione
            Li = obj.getSimulatedResidualRoute(agvId1, agvPositionMap, movingAGVId, targetNodeIdx);
            Lj = obj.getSimulatedResidualRoute(agvId2, agvPositionMap, movingAGVId, targetNodeIdx);

            if isempty(Li) || isempty(Lj) || length(Li) < 2 || length(Lj) < 2
                conflictType = 'none';
                return;
            end

            % Trova posizioni correnti nelle residual routes simulate
            currentNode1 = agvPositionMap(agvId1);
            currentNode2 = agvPositionMap(agvId2);

            ki = find(Li == currentNode1, 1);
            kj = find(Lj == currentNode2, 1);

            if isempty(ki) || isempty(kj) || ki >= length(Li) || kj >= length(Lj)
                conflictType = 'none';
                return;
            end

            % Nodi secondo Definition 7
            Tiki = Li(ki);
            Tjkj = Lj(kj);
            Tiki_plus1 = Li(ki + 1);
            Tjkj_plus1 = Lj(kj + 1);

            % Definition 7.iii: Pursuit conflict
            if Tiki_plus1 == Tjkj && ki + 2 <= length(Li) && Li(ki + 2) == Tjkj_plus1
                conflictType = 'pursuit';
            else
                conflictType = 'none';
            end
        end

        function simulatedRoute = getSimulatedResidualRoute(obj, agvId, agvPositionMap, movingAGVId, targetNodeIdx)
            % Ottiene la residual route simulata per l'AGV

            if agvId == movingAGVId
                % Per l'AGV che si sta muovendo, simula la route dal targetNodeIdx
                originalRoute = obj.AGVs(agvId).residualRoute;
                if ~isempty(originalRoute) && length(originalRoute) > 1
                    % Trova l'indice del targetNodeIdx nella route originale
                    targetIdx = find(originalRoute == targetNodeIdx, 1);
                    if ~isempty(targetIdx)
                        simulatedRoute = originalRoute(targetIdx:end);
                    else
                        % Il target node non è nella route originale (errore)
                        simulatedRoute = targetNodeIdx;
                    end
                else
                    simulatedRoute = targetNodeIdx;
                end
            else
                % Per gli altri AGV, usa la route normale ma aggiorna la posizione corrente
                originalRoute = obj.AGVs(agvId).residualRoute;
                simulatedPosition = agvPositionMap(agvId);

                if ~isempty(originalRoute)
                    % Trova l'indice della posizione simulata nella route
                    posIdx = find(originalRoute == simulatedPosition, 1);
                    if ~isempty(posIdx)
                        simulatedRoute = originalRoute(posIdx:end);
                    else
                        % La posizione simulata non è nella route (caso edge)
                        simulatedRoute = [simulatedPosition];
                    end
                else
                    simulatedRoute = [simulatedPosition];
                end
            end
        end

        function precomputeAllCycles(obj)
            % Precalcola tutti i cicli usando allcycles() di MATLAB
            %fprintf('Precalcolando tutti i cicli del grafo...\n');

            % Usa allcycles() di MATLAB per trovare tutti i cicli
            obj.allCycles = allcycles(obj.graph, "MaxCycleLength",12);

            % Filtra solo cicli che contengono nodi T/X (Definition 13)
            validCycles = {};
            for i = 1:length(obj.allCycles)
                cycle = obj.allCycles{i};
                % Verifica che tutti i nodi del ciclo siano T/X
                isValidCycle = true;
                for nodeIdx = cycle
                    nodeType = obj.nodes(nodeIdx).type;
                    if ~(strcmp(nodeType, 'T') || strcmp(nodeType, 'X'))
                        isValidCycle = false;
                        break;
                    end
                end
                if isValidCycle
                    validCycles{end+1} = cycle;
                end
            end
            obj.allCycles = validCycles;

            % Crea mappa nodo -> cicli
            obj.nodeCycleMap = containers.Map('KeyType', 'int32', 'ValueType', 'any');

            for nodeIdx = 1:length(obj.nodes)
                nodeType = obj.nodes(nodeIdx).type;
                if strcmp(nodeType, 'T') || strcmp(nodeType, 'X')
                    cycleIndices = [];
                    for cycleIdx = 1:length(obj.allCycles)
                        cycle = obj.allCycles{cycleIdx};
                        if ismember(nodeIdx, cycle)
                            cycleIndices = [cycleIndices, cycleIdx];
                        end
                    end
                    obj.nodeCycleMap(nodeIdx) = cycleIndices;
                else
                    obj.nodeCycleMap(nodeIdx) = [];
                end
            end

            %fprintf('Trovati %d cicli validi (solo nodi T/X)\n', length(obj.allCycles));
            if obj.debugMode
                obj.printCycleSummary();
            end
        end

        function printCycleSummary(obj)
            % fprintf('\n=== SOMMARIO CICLI PRECALCOLATI ===\n');
            % fprintf('Cicli totali: %d\n', length(obj.allCycles));

            for i = 1:min(10, length(obj.allCycles)) % Mostra primi 10
                cycle = obj.allCycles{i};
                nodeNames = arrayfun(@(x) obj.getNodeName(x), cycle, 'UniformOutput', false);
                %fprintf('Ciclo %d: [%s]\n', i, strjoin(nodeNames, ' → '));
            end

            if length(obj.allCycles) > 10
                %fprintf('... e altri %d cicli\n', length(obj.allCycles) - 10);
            end

            % Statistiche per nodo
            %fprintf('\nNodi con cicli:\n');
            nodeIndices = keys(obj.nodeCycleMap);
            for i = 1:length(nodeIndices)
                nodeIdx = nodeIndices{i};
                cycleIndices = obj.nodeCycleMap(nodeIdx);
                if ~isempty(cycleIndices)
                    % fprintf('  %s: appartiene a %d cicli\n', ...
                    %     obj.getNodeName(nodeIdx), length(cycleIndices));
                end
            end
            %fprintf('===============================\n\n');
        end

        function testCycleDetection(obj)
            % Test per verificare che il cycle detection funzioni
            fprintf('\n=== TEST CYCLE DETECTION ===\n');

            % Test su alcuni nodi T/X
            testNodes = [];
            for i = 1:length(obj.nodes)
                nodeType = obj.nodes(i).type;
                if strcmp(nodeType, 'T') || strcmp(nodeType, 'X')
                    testNodes = [testNodes, i];
                end
            end

            fprintf('Testando cycle pursuit-free condition su %d nodi T/X\n', length(testNodes));

            for nodeIdx = testNodes(1:min(5, length(testNodes))) % Testa primi 5
                nodeName = obj.getNodeName(nodeIdx);
                isPursuitFree = obj.checkCyclePursuitFree(nodeIdx);

                if obj.nodeCycleMap.isKey(nodeIdx)
                    cycleCount = length(obj.nodeCycleMap(nodeIdx));
                else
                    cycleCount = 0;
                end

                fprintf('  %s: %d cicli, pursuit-free = %s\n', ...
                    nodeName, cycleCount, mat2str(isPursuitFree));
            end

            fprintf('============================\n');
        end
        
        % === 3. PATH SATURATION CHECK (Rule 9) ===
        function saturated = checkPathSaturation(obj, agv1, agv2)
            % Rule 9 (Verma et al. 2024):
            % Controlla S(Ti,p+2) e S(Tk,l+1) sulle route di Ri e Rk dopo Ny.
            % CORREZIONE: se Ti,p+2 o Tk,l+1 coincide con Nx,
            % Ri sta per lasciare Nx → usare S(Nx)-1 come stato effettivo.

            saturated = false;

            route1 = obj.AGVs(agv1).residualRoute;  % [Nx, Ny, Ti,p+2, ...]
            route2 = obj.AGVs(agv2).residualRoute;  % [..., Ny, Tk,l+1, ...]

            % Eccezione (I): Nx ∈ WP → skip
            Nx = obj.AGVs(agv1).logicalNode;
            typeNx = obj.nodes(Nx).type;
            if strcmp(typeNx, 'P') || contains(typeNx, 'W')
                return;
            end

            % Servono almeno [Nx, Ny, Ti,p+2]
            if numel(route1) < 3
                return;
            end

            Ny    = route1(2);
            Ti_p2 = route1(3);

            % Tk,l+1: nodo dopo Ny nella route di Rk
            nyIdx = find(route2 == Ny, 1, 'first');
            if isempty(nyIdx) || nyIdx >= numel(route2)
                return;
            end
            Tk_l1 = route2(nyIdx + 1);

            % Eccezione (II): nodi WP esclusi dal controllo
            isWP_Ti_p2 = strcmp(obj.nodes(Ti_p2).type, 'P') || ...
                contains(obj.nodes(Ti_p2).type, 'W');
            isWP_Tk_l1 = strcmp(obj.nodes(Tk_l1).type, 'P') || ...
                contains(obj.nodes(Tk_l1).type, 'W');

            sat1 = false;
            sat2 = false;

            if ~isWP_Ti_p2
                % Se Ti,p+2 == Nx: Ri lascia Nx, quindi S(Nx) effettivo = S(Nx)-1
                s1 = obj.nodeStates(Ti_p2);
                if Ti_p2 == Nx, s1 = s1 - 1; end
                sat1 = (s1 >= 2);
            end

            if ~isWP_Tk_l1
                % Se Tk,l+1 == Nx: Ri lascia Nx, quindi S(Nx) effettivo = S(Nx)-1
                s2 = obj.nodeStates(Tk_l1);
                if Tk_l1 == Nx, s2 = s2 - 1; end
                sat2 = (s2 >= 2);
            end

            saturated = sat1 || sat2;

            if saturated
                return;
            end

            % ── (B) ADJACENT TX SATURATION nella residual route di Ri ────────────────
            % "adjacent TX-nodes in residual routes are never saturated" (Verma et al.)
            % Simula lo stato dopo che Ri lascia Nx e arriva in Ny, poi verifica
            % che nella route residuale non esistano due TX consecutivi entrambi S=2.

            % Stato simulato dopo la mossa di Ri
            sAfter = obj.nodeStates;
            sAfter(Nx) = sAfter(Nx) - 1;   % Ri lascia Nx
            sAfter(Ny) = sAfter(Ny) + 1;   % Ri arriva in Ny

            % Scorri la residual route e cerca coppie consecutive di TX entrambe sature
            for k = 1 : numel(route1) - 1
                n1 = route1(k);
                n2 = route1(k+1);

                % Considera solo nodi TX (non WP/P)
                isT1 = strcmp(obj.nodes(n1).type, 'T') || strcmp(obj.nodes(n1).type, 'X');
                isT2 = strcmp(obj.nodes(n2).type, 'T') || strcmp(obj.nodes(n2).type, 'X');

                if isT1 && isT2 && sAfter(n1) >= 2 && sAfter(n2) >= 2
                    saturated = true;
                    return;
                end
            end
        end
               
        function [hasConflict, otherAGV] = checkExitCPBlocked(obj, agvId)
            % Rule 3: l'altro AGV nel nodo corrente occupa l'exit CP di Ri?
            % L'exit CP e' calcolato GEOMETRICAMENTE (Definition 8: nearest CP di Nx
            % a Ny), non dalla tabella di rotta cache-ata, che puo' essere disallineata
            % (find-first su nodi ripetuti / currentSegmentIndex stale) e produrre
            % blocchi fantasma.
            hasConflict = false;
            otherAGV = [];

            currentNode = obj.AGVs(agvId).logicalNode;
            nextNode = obj.getNextNodeInRoute(agvId);
            if isempty(nextNode), return; end

            % Exit CP reale = CP di currentNode piu' vicino a nextNode
            exitCP = obj.findCPConnectionToNode(currentNode, nextNode);
            if isempty(exitCP), return; end

            exitCP_global = obj.getGlobalCPIndex(currentNode, exitCP);
            if obj.cpStates(exitCP_global) == 1
                occupantId = obj.cpOccupants(exitCP_global);
                if occupantId ~= agvId
                    hasConflict = true;
                    otherAGV = occupantId;
                end
            end
        end
        
        function isAt = isAGVAtCP(obj, agvId, nodeIdx, cpIdx)
            % Verifica se AGV è al CP specificato
            agv = obj.AGVs(agvId);
            isAt = (agv.logicalNode == nodeIdx && agv.logicalCP == cpIdx && ~agv.isMoving);
        end
        
        % === 5. GESTIONE MANOVRE ===
        function hasPending = hasPendingManeuver(obj, agvId)
            % Verifica se ci sono manovre pendenti per questo AGV
            hasPending = false;
            for i = 1:length(obj.pendingManeuvers)
                maneuver = obj.pendingManeuvers(i);
                % Tutti i tipi usano agv1/agv2
                if (maneuver.agv1 == agvId || maneuver.agv2 == agvId)
                    hasPending = true;
                    return;
                end
            end
        end

        function executeManeuver(obj, agvId)
            for i = length(obj.pendingManeuvers):-1:1
                maneuver = obj.pendingManeuvers(i);
                if obj.isAGVInManeuver(agvId, maneuver)
                    agv1 = maneuver.agv1;
                    agv2 = maneuver.agv2;
                    if obj.AGVs(agv1).state == 2 && obj.AGVs(agv2).state == 2 && ...
                            ~obj.AGVs(agv1).isMoving && ~obj.AGVs(agv2).isMoving

                        obj.changeAGVState(maneuver.agv1, obj.AGVs(maneuver.agv1).state, 3);
                        obj.changeAGVState(maneuver.agv2, obj.AGVs(maneuver.agv2).state, 3);

                        % Solo Res2 arriva in pending: Res1 è sempre in active direttamente.
                        maneuver.duration  = obj.res2Duration;
                        maneuver.startTime = obj.currentTime;
                        maneuver.endTime   = obj.currentTime + maneuver.duration;

                        obj.activeManeuvers  = [obj.activeManeuvers, maneuver];
                        obj.pendingManeuvers(i) = [];

                        if obj.debugMode
                            fprintf('🔧 Manovra %s avviata: AGV %d-%d (durata %.1fs)\n', ...
                                maneuver.type, agv1, agv2, maneuver.duration);
                        end
                        break;
                    end
                end
            end
        end

        function canExecute = canExecuteManeuver(obj, agvId)
            canExecute = false;
            for i = 1:length(obj.pendingManeuvers)
                maneuver = obj.pendingManeuvers(i);
                if obj.isAGVInManeuver(agvId, maneuver)
                    agv1 = maneuver.agv1;
                    agv2 = maneuver.agv2;
                    canExecute = (obj.AGVs(agv1).state == 2 && obj.AGVs(agv2).state == 2 && ...
                        ~obj.AGVs(agv1).isMoving && ~obj.AGVs(agv2).isMoving);
                    return;
                end
            end
        end

        function hasActive = hasActiveManeuver(obj, agvId)
            hasActive = false;
            for i = 1:length(obj.activeManeuvers)
                maneuver = obj.activeManeuvers(i);
                if (maneuver.agv1 == agvId || maneuver.agv2 == agvId)
                    hasActive = true;
                    return;
                end
            end
        end

        function updateActiveManeuvers(obj)
            % Promotion pass: promuovi Res2 pending pronte ad active.
            % Safety net indipendente da executeTrafficControl.
            for i = length(obj.pendingManeuvers):-1:1
                m = obj.pendingManeuvers(i);
                if obj.AGVs(m.agv1).state == 2 && obj.AGVs(m.agv2).state == 2 && ...
                        ~obj.AGVs(m.agv1).isMoving && ~obj.AGVs(m.agv2).isMoving
                    m.startTime = obj.currentTime;
                    m.endTime   = obj.currentTime + obj.res2Duration;
                    obj.changeAGVState(m.agv1, obj.AGVs(m.agv1).state, 3);
                    obj.changeAGVState(m.agv2, obj.AGVs(m.agv2).state, 3);
                    obj.activeManeuvers  = [obj.activeManeuvers, m];
                    obj.pendingManeuvers(i) = [];
                    if obj.debugMode
                        fprintf('⚡ [updateActive] Res2 promossa: AGV%d-AGV%d in %s\n', ...
                            m.agv1, m.agv2, obj.getNodeName(m.nodeIdx));
                    end
                end
            end

            for i = length(obj.activeManeuvers):-1:1
                maneuver = obj.activeManeuvers(i);

                if strcmp(maneuver.type, 'Res1')
                    % Res1: completata da updatePhysicalMovements (movimento fisico di Rk).
                    % Manteniamo lo stato resolving finché Rk è ancora in moto.
                    if obj.AGVs(maneuver.agv2).isMoving
                        % obj.AGVs(maneuver.agv1).state = 3;
                        % obj.AGVs(maneuver.agv2).state = 3;
                        obj.changeAGVState(maneuver.agv1, obj.AGVs(maneuver.agv1).state, 3);
                        obj.changeAGVState(maneuver.agv2, obj.AGVs(maneuver.agv2).state, 3);

                    end
                    % Se Rk ha smesso di muoversi, completeRes1Movement lo ha già gestito.
                else
                    % Res2: completata da timer
                    if obj.currentTime >= maneuver.endTime
                        obj.completeManeuver(maneuver, i);
                    else
                        % obj.AGVs(maneuver.agv1).state = 3;
                        % obj.AGVs(maneuver.agv2).state = 3;
                        obj.changeAGVState(maneuver.agv1, obj.AGVs(maneuver.agv1).state, 3);
                        obj.changeAGVState(maneuver.agv2, obj.AGVs(maneuver.agv2).state, 3);
                    end
                end
            end
        end

        function completeManeuver(obj, maneuver, maneuverIdx)
            if obj.debugMode
                fprintf('✅ Manovra %s completata: AGV %d-%d\n', ...
                    maneuver.type, maneuver.agv1, maneuver.agv2);
            end

            if strcmp(maneuver.type, 'Res2')
                obj.executeRes2(maneuver);
                obj.changeAGVState(maneuver.agv1, obj.AGVs(maneuver.agv1).state, 2);
                obj.changeAGVState(maneuver.agv2, obj.AGVs(maneuver.agv2).state, 2);
                obj.activeManeuvers(maneuverIdx) = [];
                % Paper Rule 4: continuazione immediata a Rule 5/7 per entrambi.
                obj.applyPostManeuverRules(maneuver.agv1);
                obj.applyPostManeuverRules(maneuver.agv2);
            else
                % Res1 non dovrebbe arrivare qui (gestita da completeRes1Movement).
                if obj.debugMode
                    fprintf('⚠️  completeManeuver: Res1 inaspettata per AGV%d-AGV%d\n', ...
                        maneuver.agv1, maneuver.agv2);
                end
                obj.executeRes1(maneuver);
                s = obj.setAGVToResuming(maneuver.agv1);
                if ~s, obj.setAGVWaiting(maneuver.agv1); end
                obj.activeManeuvers(maneuverIdx) = [];
            end
        end

        function applyPostManeuverRules(obj, agvId)
            nextNode = obj.getNextNodeInRoute(agvId);
            if isempty(nextNode)
                obj.changeAGVState(agvId, obj.AGVs(agvId).state, 0);
                obj.handleTaskCompletion(agvId);
                return;
            end
            nodeType = obj.nodes(nextNode).type;
            if strcmp(nodeType, 'T') || strcmp(nodeType, 'X')
                obj.applyRule5(agvId);
            else
                obj.applyRule7(agvId);
            end
        end
        
        function isInManeuver = isAGVInManeuver(obj, agvId, maneuver)
            % Verifica se AGV è coinvolto nella manovra
            isInManeuver = false;

            % Entrambi i tipi usano agv1/agv2
            isInManeuver = (maneuver.agv1 == agvId || maneuver.agv2 == agvId);
        end

        function executeSingleManeuver(obj, maneuver, maneuverIdx)
            if obj.debugMode
                fprintf('🔧 Eseguendo manovra %s: AGV %d-%d in nodo %s\n', ...
                    maneuver.type, maneuver.agv1, maneuver.agv2, obj.getNodeName(maneuver.nodeIdx));
            end
            % Esegue una singola manovra
            if strcmp(maneuver.type, 'Res1')
                obj.executeRes1(maneuver);
            elseif strcmp(maneuver.type, 'Res2')
                obj.executeRes2(maneuver);
            end
            
            % Rimuovi manovra completata
            obj.pendingManeuvers(maneuverIdx) = [];
            if obj.debugMode
                fprintf('✅ Manovra %s completata e rimossa\n', maneuver.type);
            end
        end
        
        function success = requestRes1(obj, requestingAGV, blockingAGV, nodeIdx)
            % Res1: nessuna fase pending — va direttamente in active.
            % blockingAGV inizia un movimento fisico intra-nodo verso targetCP.
            % entryCP rimane occupato fino all'arrivo fisico di blockingAGV.
            % La Res1 si conclude in updatePhysicalMovements quando blockingAGV arriva.

            [entryCP, exitCP] = obj.getCPsFromPrecomputedTable(requestingAGV, nodeIdx);
            targetCP = obj.findFreeCPExcluding(nodeIdx, [entryCP, exitCP]);

            if isempty(targetCP)
                % Condizione (c) del paper violata: terzo AGV occupa i CP rimanenti.
                obj.setAGVWaiting(requestingAGV);
                success = false;
                return;
            end

            % Guardia: blockingAGV deve essere fermo
            if obj.AGVs(blockingAGV).isMoving
                obj.setAGVWaiting(requestingAGV);
                success = false;
                return;
            end

            % Pre-riserva targetCP → SNy = 2
            targetCP_global = obj.getGlobalCPIndex(nodeIdx, targetCP);
            obj.cpStates(targetCP_global) = 1;
            obj.cpOccupants(targetCP_global) = blockingAGV;
            obj.nodeStates(nodeIdx) = obj.nodeStates(nodeIdx) + 1;

            % Crea maneuver e aggiungi DIRETTAMENTE ad activeManeuvers (skip pending)
            maneuver = struct();
            maneuver.type      = 'Res1';
            maneuver.agv1      = requestingAGV;
            maneuver.agv2      = blockingAGV;
            maneuver.nodeIdx   = nodeIdx;
            maneuver.targetCP  = targetCP;
            maneuver.duration  = obj.res1Duration;
            maneuver.startTime = obj.currentTime;
            maneuver.endTime   = obj.currentTime + obj.res1Duration;

            obj.activeManeuvers = [obj.activeManeuvers, maneuver];

            % Entrambi in resolving; blockingAGV inizia il movimento fisico intra-nodo
            % obj.AGVs(requestingAGV).state     = 3;  % resolving — attende Rk
            % obj.AGVs(blockingAGV).state       = 3;  % resolving — in movimento
            obj.changeAGVState(requestingAGV, obj.AGVs(requestingAGV).state, 3);
            obj.changeAGVState(blockingAGV,   obj.AGVs(blockingAGV).state,   3);
            obj.AGVs(blockingAGV).isMoving    = true;
            obj.AGVs(blockingAGV).arrivalTime = obj.currentTime + obj.res1Duration;

            % Metriche
            obj.metrics.maneuverCounts.res1               = obj.metrics.maneuverCounts.res1 + 1;
            obj.AGVs(requestingAGV).maneuverCounts.res1   = obj.AGVs(requestingAGV).maneuverCounts.res1 + 1;
            obj.AGVs(blockingAGV).maneuverCounts.res1     = obj.AGVs(blockingAGV).maneuverCounts.res1 + 1;

            if obj.debugMode
                fprintf('🔧 Res1 avviata: AGV%d→AGV%d in %s, Rk a CP%d (durata %.1fs)\n', ...
                    requestingAGV, blockingAGV, obj.getNodeName(nodeIdx), targetCP, obj.res1Duration);
            end

            success = false;  % requestingAGV rimane in resolving finché Rk non arriva
        end

        function success = requestRes2(obj, agv1, agv2)
            % Res2: diretta ad active se entrambi fisicamente presenti,
            % in pending con startTime/endTime=NaN se un AGV è ancora in transito.

            maneuver = struct();
            maneuver.type      = 'Res2';
            maneuver.agv1      = agv1;
            maneuver.agv2      = agv2;
            maneuver.nodeIdx   = obj.AGVs(agv1).logicalNode;
            maneuver.targetCP  = [];
            maneuver.duration  = obj.res2Duration;

            bothPresent = (~obj.AGVs(agv1).isMoving && ~obj.AGVs(agv2).isMoving);

            if bothPresent
                % Entrambi fermi nel nodo → active diretta, timer parte subito
                maneuver.startTime = obj.currentTime;
                maneuver.endTime   = obj.currentTime + obj.res2Duration;
                obj.activeManeuvers  = [obj.activeManeuvers, maneuver];
                obj.changeAGVState(agv1, obj.AGVs(agv1).state, 3);
                obj.changeAGVState(agv2, obj.AGVs(agv2).state, 3);
                if obj.debugMode
                    fprintf('⚡ Res2 immediata: AGV%d-AGV%d in %s (%.1fs)\n', ...
                        agv1, agv2, obj.getNodeName(maneuver.nodeIdx), obj.res2Duration);
                end
            else
                % Un AGV è ancora in transito → pending fino all'arrivo fisico
                maneuver.startTime = NaN;
                maneuver.endTime   = NaN;
                obj.pendingManeuvers = [obj.pendingManeuvers, maneuver];
                if obj.debugMode
                    fprintf('⏳ Res2 in pending: AGV%d-AGV%d in %s (AGV in transito)\n', ...
                        agv1, agv2, obj.getNodeName(maneuver.nodeIdx));
                end
            end

            obj.metrics.maneuverCounts.res2               = obj.metrics.maneuverCounts.res2 + 1;
            obj.AGVs(agv1).maneuverCounts.res2 = obj.AGVs(agv1).maneuverCounts.res2 + 1;
            obj.AGVs(agv2).maneuverCounts.res2 = obj.AGVs(agv2).maneuverCounts.res2 + 1;
            success = false;
        end
        
        function executeRes1(obj, maneuver)
            % Esegue Res1: sposta blockingAGV al targetCP pre-riservato,
            % liberando l'entryCP di requestingAGV in Ny.

            requestingAGV = maneuver.agv1;
            blockingAGV   = maneuver.agv2;
            nodeIdx       = maneuver.nodeIdx;
            targetCP      = maneuver.targetCP;  % Già pre-riservato in requestRes1

            if obj.debugMode
                fprintf('🔧 executeRes1: AGV %d richiede, AGV %d blocca in %s → sposta a CP%d\n', ...
                    requestingAGV, blockingAGV, obj.getNodeName(nodeIdx), targetCP);
            end

            currentCP = obj.AGVs(blockingAGV).logicalCP;

            if currentCP ~= targetCP
                % Rilascia il CP corrente di blockingAGV
                currentCP_global = obj.getGlobalCPIndex(nodeIdx, currentCP);
                obj.cpStates(currentCP_global)    = 0;
                obj.cpOccupants(currentCP_global) = 0;

                % nodeStates: il targetCP era già contato (+1 in requestRes1).
                % Rilasciando currentCP, torniamo da 2 a 1 — corretto.
                obj.nodeStates(nodeIdx) = obj.nodeStates(nodeIdx) - 1;

                % Il targetCP è già fisicamente riservato: aggiorna solo logicalCP
                obj.AGVs(blockingAGV).logicalCP = targetCP;

                if obj.debugMode
                    fprintf('   AGV %d spostato da CP%d a CP%d (targetCP pre-riservato)\n', ...
                        blockingAGV, currentCP, targetCP);
                end
            end

            %obj.AGVs(blockingAGV).state   = 2;  % waiting
            obj.changeAGVState(blockingAGV, obj.AGVs(blockingAGV).state, 2);
            %obj.AGVs(requestingAGV).state = 2;  % waiting — gestito da completeRes1Movement
        end
        
        function executeRes2(obj, maneuver)
            
            % Esegue manovra Res2: posiziona entrambi AGV ai loro exit CP
            agv1 = maneuver.agv1;
            agv2 = maneuver.agv2;
            nodeIdx = maneuver.nodeIdx;

            % Verifica che entrambi siano nel nodo e fermi
            if obj.AGVs(agv1).logicalNode ~= nodeIdx || obj.AGVs(agv1).isMoving || ...
                    obj.AGVs(agv2).logicalNode ~= nodeIdx || obj.AGVs(agv2).isMoving
                return;
            end

            % Calcola exit CP per entrambi usando la tabella precalcolata
            nextNode1 = obj.getNextNodeInRoute(agv1);
            nextNode2 = obj.getNextNodeInRoute(agv2);
            
            if ~isempty(nextNode1)
                [~, exitCP1] = obj.getCPsFromPrecomputedTable(agv1, nextNode1);
            else
                exitCP1 = obj.AGVs(agv1).logicalCP; % Rimane dove è
            end
            
            if ~isempty(nextNode2)
                [~, exitCP2] = obj.getCPsFromPrecomputedTable(agv2, nextNode2);
            else
                exitCP2 = obj.AGVs(agv2).logicalCP; % Rimane dove è
            end

            % CORREZIONE: gestione atomica degli stati CP
            currentCP1 = obj.AGVs(agv1).logicalCP;
            currentCP2 = obj.AGVs(agv2).logicalCP;

            % Sostituisci il blocco "Evita conflitti se exit CP uguali" (righe ~2900-2906)
            if exitCP1 == exitCP2
                % Exit CP coincidenti: Res2 non puo' separare i due AGV (situazione
                % di pursuit). NON inventare un CP alternativo: lascia tutto invariato.
                % Il guard in Rule 4 impedisce che si arrivi qui; se accade, e' un
                % segnale di problema a monte (rotte/CP table) da investigare.
                if obj.debugMode
                    fprintf('[Res2] exit CP coincidenti (CP%d) per AGV %d-%d in %s: manovra inefficace\n', ...
                        exitCP1, agv1, agv2, obj.getNodeName(nodeIdx));
                end
                return;
            end

            % Rilascia CP correnti atomicamente
            if currentCP1 ~= exitCP1
                currentCP1_global = obj.getGlobalCPIndex(nodeIdx, currentCP1);
                obj.cpStates(currentCP1_global) = 0;
                obj.cpOccupants(currentCP1_global) = 0;
            end

            if currentCP2 ~= exitCP2
                currentCP2_global = obj.getGlobalCPIndex(nodeIdx, currentCP2);
                obj.cpStates(currentCP2_global) = 0;
                obj.cpOccupants(currentCP2_global) = 0;
            end

            % Occupa nuovi CP atomicamente
            if currentCP1 ~= exitCP1
                exitCP1_global = obj.getGlobalCPIndex(nodeIdx, exitCP1);
                obj.cpStates(exitCP1_global) = 1;
                obj.cpOccupants(exitCP1_global) = agv1;
                obj.AGVs(agv1).logicalCP = exitCP1;

                if obj.debugMode
                    fprintf('AGV %d spostato in %s da CP%d a CP%d (Res2)\n', ...
                        agv1, obj.getNodeName(nodeIdx), currentCP1, exitCP1);
                end
            end

            if currentCP2 ~= exitCP2
                exitCP2_global = obj.getGlobalCPIndex(nodeIdx, exitCP2);
                obj.cpStates(exitCP2_global) = 1;
                obj.cpOccupants(exitCP2_global) = agv2;
                obj.AGVs(agv2).logicalCP = exitCP2;

                if obj.debugMode
                    fprintf('AGV %d spostato in %s da CP%d a CP%d (Res2)\n', ...
                        agv2, obj.getNodeName(nodeIdx), currentCP2, exitCP2);
                end
            end

            % obj.changeAGVState(maneuver.agv1, obj.AGVs(maneuver.agv1).state, 2);
            % obj.changeAGVState(maneuver.agv2, obj.AGVs(maneuver.agv2).state, 2);
        end
        
        % === FUNZIONI AUSILIARIE PER MANOVRE ===
        function targetCP = findFreeCPExcluding(obj, nodeIdx, excludeCPs)
            % Trova CP libero escludendo quelli specificati
            cpData = obj.controlPoints(nodeIdx);
            targetCP = [];
            
            for cp = 1:cpData.nCPs
                if ~ismember(cp, excludeCPs)
                    globalIdx = cpData.globalIndices(cp);
                    if obj.cpStates(globalIdx) == 0
                        targetCP = cp;
                        return;
                    end
                end
            end
        end
        
        function alternativeCP = findAlternativeCP(obj, nodeIdx, excludeCP)
            % Trova CP alternativo
            cpData = obj.controlPoints(nodeIdx);
            alternativeCP = [];
            
            for cp = 1:cpData.nCPs
                if cp ~= excludeCP
                    globalIdx = cpData.globalIndices(cp);
                    if obj.cpStates(globalIdx) == 0
                        alternativeCP = cp;
                        return;
                    end
                end
            end
        end
        
        function moveTo(obj, agvId, nodeIdx, targetCP)
            % Sposta AGV al CP target nello stesso nodo (manovra interna)
            currentCP = obj.AGVs(agvId).logicalCP;
            
            if currentCP == targetCP
                return; % Già nella posizione corretta
            end
            
            % Rilascia CP corrente
            currentCP_global = obj.getGlobalCPIndex(nodeIdx, currentCP);
            obj.cpStates(currentCP_global) = 0;
            obj.cpOccupants(currentCP_global) = 0;
            
            % Occupa nuovo CP
            targetCP_global = obj.getGlobalCPIndex(nodeIdx, targetCP);
            obj.cpStates(targetCP_global) = 1;
            obj.cpOccupants(targetCP_global) = agvId;
            
            % Aggiorna AGV
            obj.AGVs(agvId).logicalCP = targetCP;
            
            if obj.debugMode
                fprintf('AGV %d spostato in %s da CP%d a CP%d (manovra)\n', ...
                    agvId, obj.getNodeName(nodeIdx), currentCP, targetCP);
            end
        end
        
              
        % === 6. METRICHE E REPORTING ===
        function recordMetrics(obj)
            % Registra metriche sistema ogni secondo
            if mod(obj.currentTime, 1.0) < obj.timeStep
                % Stati AGV
                idleCount = sum([obj.AGVs.state] == 0);
                resumingCount = sum([obj.AGVs.state] == 1);
                waitingCount = sum([obj.AGVs.state] == 2);
                resolvingCount = sum([obj.AGVs.state] == 3);
                
                % Occupancy nodi
                avgOccupancy = mean(obj.nodeStates);
                maxOccupancy = max(obj.nodeStates);
                
                % Salva snapshot
                snapshot = struct();
                snapshot.time = obj.currentTime;
                snapshot.idleAGVs = idleCount;
                snapshot.resumingAGVs = resumingCount;
                snapshot.waitingAGVs = waitingCount;
                snapshot.resolvingAGVs = resolvingCount;
                snapshot.avgOccupancy = avgOccupancy;
                snapshot.maxOccupancy = maxOccupancy;
                snapshot.tasksCompleted = length(obj.completedTasks);
                snapshot.tasksInQueue = length(obj.taskQueue);
                
                if ~isfield(obj.metrics, 'snapshots')
                    obj.metrics.snapshots = [];
                end
                obj.metrics.snapshots = [obj.metrics.snapshots, snapshot];
            end
        end

        function computeFinalMetrics(obj, printReport)
            % CALCOLA METRICHE FINALI DETTAGLIATE

            if nargin < 2
                printReport = true;
            end

            % Calcola metriche finali dettagliate
            totalTime = obj.currentTime;
            tasksCompleted = length(obj.completedTasks);

            % Produttività
            obj.metrics.productivity = tasksCompleted / (totalTime / 3600); % task/ora

            % Metriche per AGV
            totalWaitingTime = sum([obj.AGVs.waitingTime]);
            avgWaitingTime = totalWaitingTime / obj.nAGV;
            totalDistance = sum([obj.AGVs.totalDistance]);
            avgDistancePerAGV = totalDistance / obj.nAGV;

            % ✅ MODIFICA: Calcola distanza media per task usando completedTasks
            avgDistancePerTaskPerAGV = [];
            for i = 1:obj.nAGV
                % Trova task completati da questo AGV
                agvTaskDistances = [];
                for j = 1:length(obj.completedTasks)
                    task = obj.completedTasks(j);
                    if isfield(task, 'assignedAGV') && task.assignedAGV == i && ...
                            isfield(task, 'totalDistance') && ~isempty(task.totalDistance)
                        agvTaskDistances = [agvTaskDistances, task.totalDistance];
                    end
                end

                if ~isempty(agvTaskDistances)
                    avgDistPerTask = mean(agvTaskDistances);
                else
                    avgDistPerTask = 0;
                end
                avgDistancePerTaskPerAGV = [avgDistancePerTaskPerAGV, avgDistPerTask];
            end

            % Media globale delle distanze medie per task degli AGV
            globalAvgDistancePerTask = mean(avgDistancePerTaskPerAGV);

            % Utilizzo sistema
            if ~isempty(obj.completedTasks)
                % Calcola tempi di ESECUZIONE (assignmentTime → completionTime)
                executionTimes = [];

                for i = 1:length(obj.completedTasks)
                    task = obj.completedTasks(i);
                    if task.assignmentTime >= 0  % Solo se assignmentTime è valido
                        executionTime = task.completionTime - task.assignmentTime;
                        executionTimes = [executionTimes, executionTime];
                    end
                end

                if ~isempty(executionTimes)
                    avgExecutionTime = mean(executionTimes);
                    maxExecutionTime = max(executionTimes);
                    minExecutionTime = min(executionTimes);
                else
                    avgExecutionTime = 0;
                    maxExecutionTime = 0;
                    minExecutionTime = 0;
                end
            else
                avgExecutionTime = 0;
                maxExecutionTime = 0;
                minExecutionTime = 0;
            end

            % Task completati per AGV
            tasksPerAGV = zeros(1, obj.nAGV);
            distancePerAGV = zeros(1, obj.nAGV);

            for i = 1:obj.nAGV
                % Conta task completati da questo AGV
                tasksPerAGV(i) = obj.AGVs(i).tasksCompleted;

                % Distanza totale percorsa da questo AGV
                distancePerAGV(i) = obj.AGVs(i).totalDistance;
            end

            % Statistiche load imbalance
            obj.metrics.finalMetrics.tasksPerAGV_mean = mean(tasksPerAGV);
            obj.metrics.finalMetrics.tasksPerAGV_std = std(tasksPerAGV);
            obj.metrics.finalMetrics.tasksPerAGV_vector = tasksPerAGV; % Per analisi dettagliate

            obj.metrics.finalMetrics.distancePerAGV_mean = mean(distancePerAGV);
            obj.metrics.finalMetrics.distancePerAGV_std = std(distancePerAGV);
            obj.metrics.finalMetrics.distancePerAGV_vector = distancePerAGV; % Per analisi dettagliate

            % Statistiche task delay nel pool
            if ~isempty(obj.taskDelaysInPool)
                obj.metrics.finalMetrics.taskDelay_mean = mean(obj.taskDelaysInPool);
                obj.metrics.finalMetrics.taskDelay_std = std(obj.taskDelaysInPool);
            else
                obj.metrics.finalMetrics.taskDelay_mean = 0;
                obj.metrics.finalMetrics.taskDelay_std = 0;
            end

            % Conflitti e manovre
            totalConflicts = obj.metrics.conflictCounts.headon + ...
                obj.metrics.conflictCounts.intersection + ...
                obj.metrics.conflictCounts.pursuit + ...
                obj.metrics.conflictCounts.loop;

            totalManeuvers = obj.metrics.maneuverCounts.res1 + ...
                obj.metrics.maneuverCounts.res2;

            if printReport
                % Report finale
                fprintf('\n=== METRICHE FINALI IDRR ===\n');
                fprintf('Tempo simulazione: %.1f s\n', totalTime);
                fprintf('Task completati: %d\n', tasksCompleted);
                fprintf('Produttività: %.2f task/ora\n', obj.metrics.productivity);
                fprintf('Tempo attesa medio per AGV: %.2f s\n', avgWaitingTime);
                fprintf('Distanza Totale media per AGV: %.1f m\n', avgDistancePerAGV);
                fprintf('Distanza media per task (media AGV): %.1f m\n', globalAvgDistancePerTask);
                fprintf('Tempo esecuzione task medio: %.2f s\n', avgExecutionTime);
                fprintf('Tempo esecuzione task max: %.2f s\n', maxExecutionTime);
                fprintf('Tempo esecuzione task min: %.2f s\n', minExecutionTime);
                fprintf('Conflitti totali: %d (head-on:%d, intersection:%d, pursuit:%d, loop:%d)\n', ...
                    totalConflicts, obj.metrics.conflictCounts.headon, ...
                    obj.metrics.conflictCounts.intersection, obj.metrics.conflictCounts.pursuit, ...
                    obj.metrics.conflictCounts.loop);
                fprintf('Manovre totali: %d (Res1:%d, Res2:%d)\n', ...
                    totalManeuvers, obj.metrics.maneuverCounts.res1, obj.metrics.maneuverCounts.res2);
                fprintf('===============================\n');
            end
            

            % Salva metriche finali
            obj.metrics.finalMetrics = struct();
            obj.metrics.finalMetrics.productivity = obj.metrics.productivity;
            obj.metrics.finalMetrics.avgWaitingTime = avgWaitingTime;
            obj.metrics.finalMetrics.avgDistancePerAGV = avgDistancePerAGV;
            obj.metrics.finalMetrics.avgDistancePerAGV = avgDistancePerAGV;
            obj.metrics.finalMetrics.globalAvgDistancePerTask = globalAvgDistancePerTask;
            obj.metrics.finalMetrics.avgDistancePerTaskPerAGV = avgDistancePerTaskPerAGV;
            obj.metrics.finalMetrics.avgExecutionTime = avgExecutionTime;
            obj.metrics.finalMetrics.maxExecutionTime = maxExecutionTime;
            obj.metrics.finalMetrics.minExecutionTime = minExecutionTime;
            % === PICKUP STATION PRODUCTIVITY ===
            pickupNodes = find(strcmp(obj.nodeTypes, 'W_pickup'));
            nPickups = length(pickupNodes);
            pickupTaskCounts = zeros(nPickups, 1);

            % Estrai task counts per ogni pickup
            for i = 1:nPickups
                nodeId = pickupNodes(i);
                if obj.pickupStationCounts.isKey(nodeId)
                    pickupTaskCounts(i) = obj.pickupStationCounts(nodeId);
                end
            end

            % Salva vettore task counts e statistiche
            obj.metrics.finalMetrics.pickupTaskCounts_vector = pickupTaskCounts;
            obj.metrics.finalMetrics.pickupTaskCounts_mean = mean(pickupTaskCounts);
            obj.metrics.finalMetrics.pickupTaskCounts_std = std(pickupTaskCounts);
            obj.metrics.finalMetrics.pickupTaskCounts_CV = ...
                (std(pickupTaskCounts) / mean(pickupTaskCounts)) * 100;
            obj.metrics.finalMetrics.totalConflicts = totalConflicts;
            obj.metrics.finalMetrics.totalManeuvers = totalManeuvers;

            % Task completati per AGV
            tasksPerAGV = zeros(1, obj.nAGV);
            distancePerAGV = zeros(1, obj.nAGV);

            for i = 1:obj.nAGV
                % Conta task completati da questo AGV
                tasksPerAGV(i) = obj.AGVs(i).tasksCompleted;

                % Distanza totale percorsa da questo AGV
                distancePerAGV(i) = obj.AGVs(i).totalDistance;
            end

            % Statistiche load imbalance
            obj.metrics.finalMetrics.tasksPerAGV_mean = mean(tasksPerAGV);
            obj.metrics.finalMetrics.tasksPerAGV_std = std(tasksPerAGV);
            obj.metrics.finalMetrics.tasksPerAGV_vector = tasksPerAGV; % Per analisi dettagliate

            obj.metrics.finalMetrics.distancePerAGV_mean = mean(distancePerAGV);
            obj.metrics.finalMetrics.distancePerAGV_std = std(distancePerAGV);
            obj.metrics.finalMetrics.distancePerAGV_vector = distancePerAGV; % Per analisi dettagliate

            % Statistiche task delay nel pool
            if ~isempty(obj.taskDelaysInPool)
                obj.metrics.finalMetrics.taskDelay_mean = mean(obj.taskDelaysInPool);
                obj.metrics.finalMetrics.taskDelay_std = std(obj.taskDelaysInPool);
            else
                obj.metrics.finalMetrics.taskDelay_mean = 0;
                obj.metrics.finalMetrics.taskDelay_std = 0;
            end
        end
        
        function printSimulationStatus(obj)
            % Status dettagliato simulazione
            nIdle = sum([obj.AGVs.state] == 0);
            nMoving = sum([obj.AGVs.state] == 1);
            nWaiting = sum([obj.AGVs.state] == 2);
            nResolving = sum([obj.AGVs.state] == 3);
            
            % Posizioni AGV
            nInParking = 0;
            nInStations = 0;
            for i = 1:obj.nAGV
                nodeType = obj.nodes(obj.AGVs(i).logicalNode).type;
                if strcmp(nodeType, 'P')
                    nInParking = nInParking + 1;
                elseif contains(nodeType, 'W')
                    nInStations = nInStations + 1;
                end
            end
            
            fprintf('Tempo: %.1fs | Task: %d coda, %d/%d completati | AGV: %d idle (%dP,%dW), %d moving, %d waiting, %d resolving\n', ...
                obj.currentTime, length(obj.taskQueue), length(obj.completedTasks), obj.tasksGenerated, ...
                nIdle, nInParking, nInStations, nMoving, nWaiting, nResolving);

            % Dopo la riga esistente con fprintf, aggiungi:
            if ~isempty(obj.pendingManeuvers) && obj.debugMode
                fprintf('   ⚠️  Manovre pendenti: %d\n', length(obj.pendingManeuvers));
                for i = 1:length(obj.pendingManeuvers)
                    m = obj.pendingManeuvers(i);
                    fprintf('      Manovra %d: %s (AGV %d-%d, nodo %s)\n', ...
                        i, m.type, m.agv1, m.agv2, obj.getNodeName(m.nodeIdx));
                end
            end

            % Lista AGV in resolving
            resolvingAGVs = find([obj.AGVs.state] == 3);
            if ~isempty(resolvingAGVs) && obj.debugMode
                fprintf('   🔄 AGV in resolving: ');
                for agv = resolvingAGVs
                    fprintf('%d(in %s) ', agv, obj.getNodeName(obj.AGVs(agv).logicalNode));
                end
                fprintf('\n');
            end

        end
        
        function complete = isSimulationComplete(obj)
            % Verifica completamento normale (condizioni diverse per PRIM vs FIFO/RL)
            
            if strcmp(obj.operatingMode, 'PRIM_MS') || strcmp(obj.operatingMode, 'PRIM_MM')
                % === CONDIZIONI PRIM ===
                
                % 1. Tutti i task iniettati
                allTasksInjected = (obj.preGenTaskIndex > length(obj.preGeneratedTasks));
                
                % 2. Pool di asta vuoto
                poolEmpty = isempty(obj.primAuctionPool);
                
                % 3. Tutte le code Qu degli AGV vuote
                allQueuesEmpty = true;
                for i = 1:obj.nAGV
                    if ~isempty(obj.AGVs(i).unexecutedTask)
                        allQueuesEmpty = false;
                        break;
                    end
                end
                
                % 4. Nessun task attivo in esecuzione (Qe = ∅ per tutti)
                noActiveTasks = true;
                for i = 1:obj.nAGV
                    if ~isempty(obj.AGVs(i).task)
                        noActiveTasks = false;
                        break;
                    end
                end
                
                % 5. Tutti gli AGV idle in parcheggio
                allAGVsIdleInParking = true;
                for i = 1:obj.nAGV
                    agv = obj.AGVs(i);
                    if agv.state ~= 0
                        allAGVsIdleInParking = false;
                        break;
                    else
                        nodeType = obj.nodes(agv.logicalNode).type;
                        if ~strcmp(nodeType, 'P')
                            allAGVsIdleInParking = false;
                            break;
                        end
                    end
                end
                
                normalComplete = allTasksInjected && poolEmpty && allQueuesEmpty && ...
                                noActiveTasks && allAGVsIdleInParking;
                
            else
                % === CONDIZIONI FIFO/RL (originali) ===
                
                if ~isempty(obj.preGeneratedTasks)
                    allTasksGenerated = (obj.preGenTaskIndex >= length(obj.preGeneratedTasks));
                else
                    allTasksGenerated = (obj.tasksGenerated >= obj.maxTasks);
                end
                noTasksInQueue = isempty(obj.taskQueue);
                
                allAGVsIdleInParking = true;
                for i = 1:obj.nAGV
                    agv = obj.AGVs(i);
                    if agv.state ~= 0
                        allAGVsIdleInParking = false;
                        break;
                    else
                        nodeType = obj.nodes(agv.logicalNode).type;
                        if ~strcmp(nodeType, 'P')
                            allAGVsIdleInParking = false;
                            break;
                        end
                    end
                end
                
                normalComplete = allTasksGenerated && noTasksInQueue && allAGVsIdleInParking;
            end
            
            % Timeout (comune a tutte le modalità)
            timedOut = obj.currentTime >= obj.maxTime;
            
            complete = normalComplete || timedOut;
            
            % GESTIONE NOTIFICHE ALL'ENVIRONMENT
            if complete && ~isempty(obj.environmentInterface)
                if normalComplete
                    % Episodio completato normalmente
                    obj.environmentInterface.notifyEpisodeComplete(false);
                else
                    % Episodio terminato per timeout - passa flag run out
                    obj.environmentInterface.notifyEpisodeComplete(true);
                end
            end
        end
        
             
        function printSystemSummary(obj)
            % Stampa sommario configurazione sistema
            fprintf('\n=== SOMMARIO SISTEMA IDRR v2 ===\n');
            fprintf('AGV: %d\n', obj.nAGV);
            fprintf('Nodi simbolici: %d\n', length(obj.nodes));
            fprintf('  - Parcheggi (P): %d\n', sum(strcmp(obj.nodeTypes, 'P')));
            fprintf('  - Pickup (W): %d\n', sum(strcmp(obj.nodeTypes, 'W_pickup')));
            fprintf('  - Dropoff (W): %d\n', sum(strcmp(obj.nodeTypes, 'W_dropoff')));
            fprintf('  - Intersezioni (T): %d\n', sum(strcmp(obj.nodeTypes, 'T')));
            fprintf('  - Intersezioni (X): %d\n', sum(strcmp(obj.nodeTypes, 'X')));
            fprintf('Control Points: %d\n', size(obj.cpPositions, 1));
            fprintf('Route simboliche: %d\n', length(keys(obj.symbolicRoutingTable)));
            fprintf('Route geometriche: %d\n', length(keys(obj.cpRoutingTable)));
            fprintf('Velocità AGV: %.1f m/s\n', obj.agvSpeed);
            fprintf('Time step: %.1f s\n', obj.timeStep);
            fprintf('================================\n');
        end
                  
        % === METODI COMPATIBILITÀ CON CODICE ORIGINALE ===
        function visualizeSystemFinal(obj)
            % Alias per compatibilità con codice originale
            obj.visualizeCombinedLayers();
        end

        % === METODI PER INTEGRAZIONE RL ===

        function updateTop5TaskQueue(obj)
            if isempty(obj.taskQueue)
                obj.top5TaskQueue = [];
            else
                numTasks = min(5, length(obj.taskQueue));
                newTop5 = obj.taskQueue(1:numTasks);

                % Traccia ingresso di nuovi task nel pool top-5
                for i = 1:length(newTop5)
                    taskId = newTop5(i).sequentialId;
                    if ~obj.taskPoolEntryTimes.isKey(taskId)
                        % Primo ingresso di questo task nel pool
                        obj.taskPoolEntryTimes(taskId) = obj.currentTime;
                    end
                end

                obj.top5TaskQueue = newTop5;
            end
        end

        function queue = getTop5TaskQueue(obj)
            queue = obj.top5TaskQueue;
        end

        function resetThroughputTracking(obj)
            obj.lastCompletedTaskTime = 0;
            obj.lastThroughputValue = 0;
            obj.currentThroughputValue = 0;
            obj.throughputDerivative = 0;
        end

        function updateThroughputOnTaskCompletion(obj, agvId)
            obj.episodeTasksCompleted = obj.episodeTasksCompleted + 1;

            % Calcola throughput attuale
            episodeDuration = obj.currentTime - obj.episodeStartTime;
            if episodeDuration > 0
                obj.currentThroughputValue = obj.episodeTasksCompleted / (episodeDuration / 3600);
            else
                obj.currentThroughputValue = 0;
            end

            % Calcola derivata semplice
            %if obj.lastCompletedTaskTime > 0
            obj.throughputDerivative = obj.currentThroughputValue - obj.lastThroughputValue;
            % else
            %     obj.throughputDerivative = 0;
            % end

            % Aggiorna valori precedenti
            obj.lastThroughputValue = obj.currentThroughputValue;
            obj.lastCompletedTaskTime = obj.currentTime;

            obj.logMessage(sprintf('Throughput: %.2f task/h, Derivata: %.3f', ...
                obj.currentThroughputValue, obj.throughputDerivative));
        end

        function derivative = getThroughputDerivative(obj)
            derivative = obj.throughputDerivative;
        end

        function requestRLDecision(obj, agvId)
            obj.pendingRLDecisions(end+1) = agvId;
            obj.rlDecisionTimestamps(end+1) = obj.currentTime;
            obj.logMessage(sprintf('Richiesta RL per AGV %d', agvId));

            if ~isempty(obj.environmentInterface)
                try
                    obj.environmentInterface.handleDecisionRequest(agvId, obj);
                catch ME
                    obj.logMessage(sprintf('Errore environment: %s', ME.message));
                    obj.assignTaskIDRR(agvId);
                end
            end
        end

        function success = executeRLDecision(obj, agvId, selectedPosition)
            success = false;

            if selectedPosition < 1 || selectedPosition > length(obj.top5TaskQueue)
                obj.logMessage(sprintf('Posizione non valida: %d', selectedPosition));
                obj.assignTaskIDRR(agvId);
                return;
            end

            selectedTask = obj.top5TaskQueue(selectedPosition);
            obj.removePendingDecision(agvId);
            obj.resetCurrentTaskStateTimes(agvId);

            % Usa logica assegnazione esistente
            route = obj.planRouteForTaskSegmented(agvId, selectedTask);
            if ~isempty(route)
                obj.AGVs(agvId).task = selectedTask;
                obj.AGVs(agvId).residualRoute = route;
                obj.AGVs(agvId).task.assignmentTime = obj.currentTime;
                pathToPickup = obj.getOptimalPath(obj.AGVs(agvId).logicalNode, selectedTask.pickup);
                pathToDropoff = obj.getOptimalPath(selectedTask.pickup, selectedTask.dropoff);
                distToPickup = obj.calculatePathDistance(pathToPickup);
                distToDropoff = obj.calculatePathDistance(pathToDropoff);
                obj.AGVs(agvId).totalPlannedDistance = distToPickup + distToDropoff;
                obj.removeTaskFromQueues(selectedTask);
                obj.changeAGVState(agvId, 0, 2);
                obj.logMessage(sprintf('AGV %d → Task pos %d (RL)', agvId, selectedPosition));
                obj.updateTop5TaskQueue();
                success = true;
            else
                obj.assignTaskIDRR(agvId);
            end
        end

        function changeAGVState(obj, agvId, oldState, newState)
            % Hook per tracking stati
            obj.logAGVStateChange(agvId, oldState, newState);

            % Cambia stato effettivo
            obj.AGVs(agvId).state = newState;
        end

        function logAGVStateChange(obj, agvId, oldState, newState)
            % Aggiorna tempo nello stato precedente
            if oldState >= 0
                stateNames = {'Idle', 'Resuming', 'Waiting', 'Resolving'};
                if oldState+1 <= length(stateNames)
                    stateName = stateNames{oldState+1};
                    timeInState = obj.currentTime - obj.agvStateTimes(agvId).lastTransition;
                    obj.agvStateTimes(agvId).(stateName) = obj.agvStateTimes(agvId).(stateName) + timeInState;
                
                    if ~isempty(obj.AGVs(agvId).task)
                        obj.AGVs(agvId).currentTaskStateTimes.(stateName) = ...
                            obj.AGVs(agvId).currentTaskStateTimes.(stateName) + timeInState;
                    end
                end
            end

            % Crea struct corretto per state change
            if newState >= 0
                stateChange = struct();
                stateChange.timestamp = obj.currentTime;
                stateChange.oldState = oldState;
                stateChange.newState = newState;
                stateChange.duration = obj.currentTime - obj.agvStateTimes(agvId).lastTransition;

                % Aggiungi a cell array correttamente
                if isempty(obj.agvStateHistory{agvId})
                    obj.agvStateHistory{agvId} = stateChange;
                else
                    obj.agvStateHistory{agvId}(end+1) = stateChange;
                end
            end

            % Aggiorna timestamp e stato corrente
            obj.agvStateTimes(agvId).lastTransition = obj.currentTime;
            obj.agvStateTimes(agvId).currentState = newState;
        end

        function assignTaskIDRR(obj, agvId)
            % Fallback per assegnazione IDRR quando RL fallisce

            if ~isempty(obj.taskQueue) && obj.AGVs(agvId).state == 0
                % Usa logica IDRR standard
                task = obj.taskQueue(1);
                route = obj.planRouteForTaskSegmented(agvId, task);

                if ~isempty(route)
                    obj.AGVs(agvId).task = task;
                    obj.AGVs(agvId).residualRoute = route;
                    obj.AGVs(agvId).task.assignmentTime = obj.currentTime;

                    obj.taskQueue(1) = [];
                    obj.changeAGVState(agvId, obj.AGVs(agvId).state, 2); % waiting

                    obj.logMessage(sprintf('✅ Fallback IDRR: Task %d → AGV %d', ...
                        task.sequentialId, agvId));
                else
                    obj.logMessage(sprintf('⚠️ Nessun percorso valido per AGV %d', agvId));
                end
            else
                obj.logMessage(sprintf('⚠️ Fallback impossibile per AGV %d (stato=%d, tasks=%d)', ...
                    agvId, obj.AGVs(agvId).state, length(obj.taskQueue)));
            end
        end

        % METODO: resetSystem - Reset completo del sistema
        
        function resetSystem(obj)
            % Reset completo del sistema per nuovo episodio

            % Reset tempo di simulazione
            obj.currentTime = 0;
            obj.lastTaskTime = 0;
            obj.tasksGenerated = 0;
            obj.taskCounter = 0;

            obj.nodeStates(:) = 0;      % Tutti i nodi vuoti

            % Reset stati AGV e campi RL
            for i = 1:length(obj.AGVs)
                obj.AGVs(i).state = 0;              % Idle
                obj.AGVs(i).logicalNode = -1;       
                obj.AGVs(i).logicalCP = -1;
                obj.AGVs(i).tasksCompleted = 0;
                obj.AGVs(i).currentRoute = [];
                obj.AGVs(i).isMoving = false;
                obj.AGVs(i).arrivalTime = 0;
                obj.AGVs(i).totalDistance = 0;
                obj.AGVs(i).taskDistance = 0;
                obj.AGVs(i).residualRoute = [];
                obj.AGVs(i).sharedRoute = [];
                obj.AGVs(i).routeKeys = {};
                obj.AGVs(i).currentSegmentIndex = 0;
                obj.AGVs(i).finalTarget = 0;
                obj.AGVs(i).waitingTime = 0;
                obj.AGVs(i).totalPlannedDistance = 0;
                obj.AGVs(i).unexecutedTask = [];

                % Campi per tracking conflitti
                obj.AGVs(i).conflictsGenerated = 0;
                obj.AGVs(i).conflictsCumulative = 0;
                obj.AGVs(i).currentTaskStartTime = 0;
                obj.AGVs(i).currentTaskId = -1;
            end

            % Reset stati nodi e CP
            if ~isempty(obj.cpStates)
                obj.cpStates(:) = 0;                % Tutti CP liberi
            end
            if ~isempty(obj.cpOccupants)
                obj.cpOccupants(:) = 0;             % Nessun occupante
            end

            % Reset code e contatori
            obj.taskQueue = [];
            obj.completedTasks = [];

            % Reset manovre attive se esistono
            if isprop(obj, 'activeManeuvers') || isfield(obj, 'activeManeuvers')
                obj.activeManeuvers   = [];
            end

            % Stesso fix per pendingManeuvers se presente lo stesso errore
            if isprop(obj, 'pendingManeuvers') || isfield(obj, 'pendingManeuvers')
                obj.pendingManeuvers = [];
            end

            % Reset tracking task delay
            if ~isempty(obj.taskPoolEntryTimes)
                obj.taskPoolEntryTimes = containers.Map('KeyType', 'double', 'ValueType', 'double');
            end
            obj.taskDelaysInPool = [];

            % Reset contatori pickup station
            if ~isempty(obj.pickupStationCounts)
                pickupNodes = find(strcmp(obj.nodeTypes, 'W_pickup'));
                for i = 1:length(pickupNodes)
                    obj.pickupStationCounts(pickupNodes(i)) = 0;
                end
            end

            % Reset strutture PRIM
            obj.primAuctionPool = [];
            obj.preGenTaskIndex = 1;
            obj.lastInjectionTime = 0;

            obj.initializeMetrics();

            if obj.debugMode
                fprintf('🔄 Sistema IDRR completamente resetted\n');
            end
        end

        function resetForNewEpisode(obj)
            obj.logMessage('=== RESET NUOVO EPISODIO ===');

            obj.episodeStartTime = obj.currentTime;
            obj.episodeTasksCompleted = 0;
            obj.resetThroughputTracking();
            obj.deadlockCounter = 0;

            % Reset tracking
            for i = 1:obj.nAGV
                obj.agvStateHistory{i} = struct.empty();
                obj.agvStateTimes(i) = struct('Idle', 0, 'Resuming', 0, 'Waiting', 0, 'Resolving', 0, ...
                    'lastTransition', obj.currentTime, 'currentState', obj.AGVs(i).state);
            end

            for i = 1:obj.nAGV
                obj.AGVs(i).maneuverCounts = struct('res1', 0, 'res2', 0);
                obj.AGVs(i).currentTaskStateTimes = struct(...
                    'Idle', 0, 'Resuming', 0, 'Waiting', 0, 'Resolving', 0, ...
                    'taskStartTime', obj.currentTime);
            end

            obj.top5TaskQueue = [];
            obj.pendingRLDecisions = [];
            obj.rlDecisionTimestamps = [];

            obj.logMessage('Sistema reset completato');
        end

        function resetCurrentTaskStateTimes(obj, agvId)
            % Reset tempi stati per nuovo task
            obj.AGVs(agvId).currentTaskStateTimes.Idle = 0;
            obj.AGVs(agvId).currentTaskStateTimes.Resuming = 0;
            obj.AGVs(agvId).currentTaskStateTimes.Waiting = 0;
            obj.AGVs(agvId).currentTaskStateTimes.Resolving = 0;
            obj.AGVs(agvId).currentTaskStateTimes.taskStartTime = obj.currentTime;
            obj.AGVs(agvId).taskDistance = 0;

            if obj.verboseLogging
                obj.logMessage(sprintf('🔄 Reset task state times per AGV %d', agvId));
            end
        end

        function metrics = collectEpisodeMetrics(obj)
            obj.logMessage('=== COLLEZIONE METRICHE ===');

            % Finalizza tracking stati correnti
            obj.finalizeStateTracking();

            % Calcola metriche parent (riutilizza implementazione esistente)
            obj.computeFinalMetrics(false);
            parentMetrics = obj.metrics;

            % Costruisci metriche complete
            metrics = struct();

            % Metriche base (da parent)
            metrics.productivity = parentMetrics.productivity;
            metrics.conflictCounts = parentMetrics.conflictCounts;
            metrics.maneuverCounts = parentMetrics.maneuverCounts;

            % Metriche episodio
            episodeDuration = obj.currentTime - obj.episodeStartTime;
            metrics.episode = struct();
            metrics.episode.duration = episodeDuration;
            metrics.episode.tasksCompleted = obj.episodeTasksCompleted;
            metrics.episode.finalThroughput = obj.currentThroughputValue;
            metrics.episode.finalThroughputDerivative = obj.throughputDerivative;

            % Distanze (da parent)
            metrics.distances = struct();
            totalDistance = sum([obj.AGVs.totalDistance]);
            metrics.distances.total = totalDistance;
            metrics.distances.average = totalDistance / obj.nAGV;

            % Tempi stati AGV
            metrics.agvStates = struct();
            totalWaitingTime = sum([obj.AGVs.waitingTime]);
            metrics.agvStates.totalWaiting = totalWaitingTime;
            metrics.agvStates.averageWaiting = totalWaitingTime / obj.nAGV;
            metrics.agvStates.detailedTimes = obj.agvStateTimes;

            % Task execution (da parent completedTasks)
            metrics.taskExecution = struct();
            if ~isempty(obj.completedTasks)
                executionTimes = [];
                for i = 1:length(obj.completedTasks)
                    task = obj.completedTasks(i);
                    if isfield(task, 'assignmentTime') && task.assignmentTime > 0
                        execTime = task.completionTime - task.assignmentTime;
                        executionTimes(end+1) = execTime;
                    end
                end

                if ~isempty(executionTimes)
                    metrics.taskExecution.average = mean(executionTimes);
                    metrics.taskExecution.min = min(executionTimes);
                    metrics.taskExecution.max = max(executionTimes);
                end
            end

            obj.logMessage(sprintf('Metriche raccolte: %d task, %.1f min, %.2f prod', ...
                obj.episodeTasksCompleted, episodeDuration/60, metrics.productivity));
        end

        function finalizeStateTracking(obj)
            for agvId = 1:obj.nAGV
                obj.logAGVStateChange(agvId, obj.agvStateTimes(agvId).currentState, -1);
            end
        end

        function removePendingDecision(obj, agvId)
            idx = find(obj.pendingRLDecisions == agvId, 1);
            if ~isempty(idx)
                obj.pendingRLDecisions(idx) = [];
                obj.rlDecisionTimestamps(idx) = [];
            end
        end

        function removeTaskFromQueues(obj, task)
            % Rimuovi da coda principale
            for i = length(obj.taskQueue):-1:1
                if isequal(obj.taskQueue(i), task)
                    obj.taskQueue(i) = [];
                    break;
                end
            end

            % Rimuovi da top-5
            for i = length(obj.top5TaskQueue):-1:1
                if isequal(obj.top5TaskQueue(i), task)
                    obj.top5TaskQueue(i) = [];
                    break;
                end
            end
        end

        function state = getCurrentSystemState(obj)
            state = struct();
            state.currentTime = obj.currentTime;
            state.agvStates = [obj.AGVs.state];
            state.top5Queue = obj.top5TaskQueue;
            state.throughput = obj.currentThroughputValue;
            state.throughputDerivative = obj.throughputDerivative;
        end

        function logMessage(obj, message)
            if obj.verboseLogging
                timestamp = datestr(now, 'HH:MM:SS');
                fprintf('[%s] %s\n', timestamp, message);
            end
        end



        % ========== METODI PER VISUALIZZAZIONE ==========

        function initializeVisualization(obj, enableVideoRecording)
            % Inizializza visualizzazione con opzione video recording
            if nargin < 2
                enableVideoRecording = false;
            end

            try
                if enableVideoRecording
                    % Verifica disponibilità video recorder
                    if exist('IDRRVisualizer_VideoRecorder', 'class') ~= 8
                        warning('IDRRVisualizer_VideoRecorder non trovato. Usando visualizzatore standard.');
                        enableVideoRecording = false;
                    else
                        obj.visualizer = IDRRVisualizer_VideoRecorder(obj);
                        fprintf('✅ Visualizzatore con VIDEO RECORDING inizializzato\n');
                        obj.visualizationEnabled = true;
                        return;
                    end
                end

                % Fallback a visualizzatore smooth o standard
                if exist('IDRRVisualizer', 'class') ~= 8
                    if exist('IDRRVisualizer', 'class') ~= 8
                        warning('Nessun visualizzatore disponibile.');
                        obj.visualizationEnabled = false;
                        return;
                    else
                        obj.visualizer = IDRRVisualizer(obj);
                        fprintf('⚠️  Usando visualizzatore standard (possibile flickering)\n');
                    end
                else
                    obj.visualizer = IDRRVisualizer(obj);
                    fprintf('✅ Visualizzatore inizializzato\n');
                end

                obj.visualizationEnabled = true;

            catch ME
                warning(['❌ Errore inizializzazione visualizzatore: %s', ME.message]);
                obj.visualizationEnabled = false;
                obj.visualizer = [];
            end
        end

        function startVisualization(obj)
            % Avvia visualizzazione (chiamare dopo l'inizializzazione)
            if obj.visualizationEnabled && ~isempty(obj.visualizer)
                obj.visualizer.startVisualization();
            else
                warning('Visualizzatore non disponibile. Inizializza con enableVisualization=true');
            end
        end

        function stopVisualization(obj)
            % Ferma visualizzazione
            if obj.visualizationEnabled && ~isempty(obj.visualizer)
                obj.visualizer.stopVisualization();
            end
        end

        function updateVisualization(obj)
            % Aggiorna visualizzazione (chiamata automatica durante simulazione)
            if obj.visualizationEnabled && ~isempty(obj.visualizer) && isvalid(obj.visualizer)
                try
                    obj.visualizer.updateVisualization();
                catch ME
                    % Se c'è un errore nella visualizzazione, disabilitala per evitare crash
                    warning(['Errore aggiornamento visualizzazione: %s. Disabilitando...', ME.message]);
                    obj.visualizationEnabled = false;
                end
            end
        end

        function toggleVisualizationRoutes(obj)
            % Toggle visualizzazione percorsi - NOME CORRETTO
            if obj.visualizationEnabled && ~isempty(obj.visualizer)
                obj.visualizer.toggleRoutes();  % Chiama metodo corretto
            else
                fprintf('❌ Visualizzatore non disponibile\n');
            end
        end

        function toggleVisualizationConflicts(obj)
            % Toggle visualizzazione conflitti - NOME CORRETTO
            if obj.visualizationEnabled && ~isempty(obj.visualizer)
                obj.visualizer.toggleConflicts();  % Chiama metodo corretto
            else
                fprintf('❌ Visualizzatore non disponibile\n');
            end
        end

        function toggleVisualizationNodeStates(obj)
            % Toggle visualizzazione stati nodi - NOME CORRETTO
            if obj.visualizationEnabled && ~isempty(obj.visualizer)
                obj.visualizer.toggleNodeStates();  % Chiama metodo corretto
            else
                fprintf('❌ Visualizzatore non disponibile\n');
            end
        end

        function toggleVisualizationLegend(obj)
            % Toggle visualizzazione legenda
            if obj.visualizationEnabled && ~isempty(obj.visualizer)
                obj.visualizer.toggleLegend();
            else
                fprintf('❌ Visualizzatore non disponibile\n');
            end
        end

        function setVisualizationUpdateRate(obj, interval)
            % Imposta frequenza aggiornamento visualizzazione - NOME CORRETTO
            if obj.visualizationEnabled && ~isempty(obj.visualizer)
                obj.visualizer.setUpdateInterval(interval);
            else
                fprintf('❌ Visualizzatore non disponibile\n');
            end
        end

        function closeVisualization(obj)
            % Chiude finestra visualizzazione
            if obj.visualizationEnabled && ~isempty(obj.visualizer)
                obj.visualizer.close();
                obj.visualizationEnabled = false;
                obj.visualizer = [];
                fprintf('✅ Visualizzazione chiusa\n');
            else
                fprintf('ℹ️  Nessuna visualizzazione attiva da chiudere\n');
            end
        end

        function setVisualizationSmoothMode(obj, enableSmooth)
            % Abilita/disabilita modalità smooth
            if obj.visualizationEnabled && ~isempty(obj.visualizer)
                if enableSmooth
                    if ~isa(obj.visualizer, 'IDRRVisualizer')
                        obj.closeVisualization();
                        obj.visualizer = IDRRVisualizer(obj);
                        obj.visualizationEnabled = true;
                        obj.startVisualization();
                    else
                        fprintf('✅ Già in modalità SMOOTH\n');
                    end
                else
                    if ~isa(obj.visualizer, 'IDRRVisualizer')
                        fprintf('🔄 Passaggio a visualizzatore NORMALE...\n');
                        obj.closeVisualization();
                        obj.visualizer = IDRRVisualizer(obj);
                        obj.visualizationEnabled = true;
                        obj.startVisualization();
                    else
                        fprintf('✅ Già in modalità NORMALE\n');
                    end
                end
            else
                fprintf('❌ Visualizzatore non disponibile\n');
            end
        end

        function delete(obj)
            % Distruttore - pulisce risorse
            if obj.visualizationEnabled && ~isempty(obj.visualizer)
                obj.closeVisualization();
            end
        end

        % === METODI PER VIDEO RECORDING ===
        function startVideoRecording(obj, filename, quality)
            % Avvia registrazione video
            if nargin < 2 || isempty(filename)
                filename = sprintf('IDRR_Simulation_%s.mp4', datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
            end
            if nargin < 3
                quality = 'presentation'; % 'high', 'presentation', 'fast'
            end

            if ~obj.visualizationEnabled || isempty(obj.visualizer)
                fprintf('❌ Visualizzazione non abilitata\n');
                return;
            end

            if ~isa(obj.visualizer, 'IDRRVisualizer_VideoRecorder')
                fprintf('❌ Video recording non disponibile con questo visualizzatore\n');
                fprintf('💡 Usa: AGVSystem_IDRR(nAGV, maxTime, maxTasks, rate, true, true)\n');
                return;
            end

            % Configura qualità
            switch lower(quality)
                case 'high'
                    obj.visualizer.configureForHighQuality();
                case 'presentation'
                    obj.visualizer.configureForPresentation();
                case 'fast'
                    obj.visualizer.configureForFastCapture();
                otherwise
                    fprintf('⚠️  Qualità non riconosciuta, uso presentation\n');
                    obj.visualizer.configureForPresentation();
            end

            obj.visualizer.startVideoRecording(filename);
        end

        function stopVideoRecording(obj)
            % Ferma registrazione video
            if obj.visualizationEnabled && isa(obj.visualizer, 'IDRRVisualizer_VideoRecorder')
                obj.visualizer.stopVideoRecording();
            else
                fprintf('❌ Video recording non attivo\n');
            end
        end

        function setVideoQuality(obj, format, quality, frameRate, resolution)
            % Configura parametri video
            if ~obj.visualizationEnabled || ~isa(obj.visualizer, 'IDRRVisualizer_VideoRecorder')
                fprintf('❌ Video recording non disponibile\n');
                return;
            end

            if nargin >= 2 && ~isempty(format)
                obj.visualizer.setVideoFormat(format);
            end
            if nargin >= 3 && ~isempty(quality)
                obj.visualizer.setVideoQuality(quality);
            end
            if nargin >= 4 && ~isempty(frameRate)
                obj.visualizer.setVideoFrameRate(frameRate);
            end
            if nargin >= 5 && ~isempty(resolution)
                obj.visualizer.setVideoResolution(resolution(1), resolution(2));
            end
        end

        function printVideoStatus(obj)
            % Mostra stato video recording
            if obj.visualizationEnabled && isa(obj.visualizer, 'IDRRVisualizer_VideoRecorder')
                obj.visualizer.printVideoStatus();
            else
                fprintf('❌ Video recording non disponibile\n');
            end
        end

        function runSimulationWithVideo(obj, videoFilename, videoQuality)
            % Esegue simulazione completa con registrazione video automatica
            if nargin < 2
                videoFilename = sprintf('IDRR_Complete_%dAGV_%s.mp4', ...
                    obj.nAGV, datestr(now, 'yyyy-mm-dd_HH-MM'));
            end
            if nargin < 3
                videoQuality = 'presentation';
            end

            fprintf('🎬 SIMULAZIONE CON VIDEO RECORDING\n');
            fprintf('==================================\n');
            fprintf('Video: %s\n', videoFilename);
            fprintf('Qualità: %s\n', videoQuality);

            % Avvia visualizzazione se non già attiva
            if obj.visualizationEnabled
                obj.startVisualization();
            end

            % Avvia registrazione
            obj.startVideoRecording(videoFilename, videoQuality);

            % Pausa per stabilizzazione
            pause(0.2);

            % Esegui simulazione normale
            obj.runSimulation();

            % La registrazione si ferma automaticamente se autoStopAtEnd = true
            % Altrimenti fermala manualmente
            if obj.visualizationEnabled && isa(obj.visualizer, 'IDRRVisualizer_VideoRecorder')
                if ~obj.visualizer.autoStopAtEnd
                    obj.stopVideoRecording();
                end
            end

            fprintf('✅ Simulazione con video completata!\n');
            fprintf('📹 Video salvato: %s\n', videoFilename);
        end
       
        function switchToVideoMode(obj)
            % Passa da visualizzatore normale a video recorder
            if ~obj.visualizationEnabled
                fprintf('❌ Visualizzazione non attiva\n');
                return;
            end

            if isa(obj.visualizer, 'IDRRVisualizer_VideoRecorder')
                fprintf('✅ Già in modalità video recording\n');
                return;
            end

            fprintf('🔄 Passaggio a modalità video recording...\n');

            % Salva stato
            wasActive = obj.visualizer.isActive;

            % Chiudi visualizzatore corrente
            obj.closeVisualization();

            % Crea nuovo visualizzatore video
            obj.visualizer = IDRRVisualizer_VideoRecorder(obj);
            obj.visualizationEnabled = true;

            % Riattiva se era attivo
            if wasActive
                obj.startVisualization();
            end

            %fprintf('✅ Modalità video recording attivata\n');
        end
    end
end