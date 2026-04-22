%% Parameters

%% Environmental parameters
SCALE = 2;                                                                  % SCALE determines how much region to cover 
                                                                            % within the city, 2 means the whole city
NR_PER_LOC = 20;                                                            % The number of candidate perturbed locations
SAMPLE_SIZE_PPR = 1000;                                                     % The sample size when measuring perturbation   
                                                                            % probability ratio (PPR). 
NR_TEST = 5;                                                                % The number of repeated tests 
NR_VIO_SAMPLE = 1000; 
EPSILON = 0.4;                                                              % Default privacy budget

EPSILON_MAX = 3; 
epsilon_value=[0.5,1,1.5];

cell_size = [4.77, 6.87, 6.78];                                             % The cell size of the three cities are 3.18km X 3.18km
                                                                            % 4.58km X 4.58km, 4.52km X 4.52km, respectively 

%% Parameters for AIPO
NR_EPSILON_INTERVAL = 5; 

%% Parameters for COPT
LAMBDA = 5;
R = 50;
GRID_SIZE_COPT = 250;                                                        % Default value = 250; 

%% Parameters for Linear Programming based method
GRID_SIZE_LP = 18;                                                           % Default value = 12; 
GRID_SIZE_LP_LB = 12; 

