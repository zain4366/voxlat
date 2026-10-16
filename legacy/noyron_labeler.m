% =========================================================================
% NOYRON 2.0 - PHASE 5.2: AUTONOMOUS LABELING
% Batch Physics Simulation for Machine Learning
% =========================================================================

workspace_dir = 'D:\nyron 2\noyron_workspace';
input_csv = fullfile(workspace_dir, 'boom_dataset.csv');
output_csv = fullfile(workspace_dir, 'labeled_dataset.csv');

disp('Loading Noyron Data Factory dataset...');
data = readtable(input_csv);

% Initialize a new column for the Labels (Max Deflection)
data.Max_Deflection_mm = zeros(height(data), 1);

% --- CONSTANT MATERIAL PROPERTIES (Carbon Fiber) ---
rho = 1500; % Density (kg/m^3)
E = 70e9;   % Young's Modulus (Pa)
L = 0.15;   % Boom length (m)
Ro = 0.02;  % Outer radius (m)
r_hole = 0.008; % Hole radius (m)

disp('Initiating Batch Physics Simulation...');
tic; % Start a timer

for i = 1:height(data)
    % 1. Extract geometric features from the CSV
    t_mm = data.Wall_Thickness_mm(i);
    num_holes = data.Num_Holes(i);

    t = t_mm / 1000; % Convert to meters
    Ri = Ro - t;     % Inner radius

    % ----------------------------------------------------
    % 2. ANALYTICAL MASS CALCULATION
    % ----------------------------------------------------
    V_tube = pi * (Ro^2 - Ri^2) * L;
    % Each hole punches through two walls, removing two cylinders of material
    V_hole = num_holes * 2 * (pi * r_hole^2 * t);
    V_total = V_tube - V_hole;

    m = V_total * rho; % Mass = Volume * Density

    % ----------------------------------------------------
    % 3. ANALYTICAL STIFFNESS CALCULATION
    % ----------------------------------------------------
    % Area Moment of Inertia for a hollow tube
    I = (pi/4) * (Ro^4 - Ri^4);
    % Apply a structural penalty: holes reduce bending stiffness
    I_effective = I * (1 - (0.02 * num_holes));

    k = (3 * E * I_effective) / (L^3); % k = 3EI / L^3

    % ----------------------------------------------------
    % 4. DYNAMIC SIMULATION (The SDOF Math)
    % ----------------------------------------------------
    zeta = 0.02; % 2% Damping ratio
    c = 2 * zeta * sqrt(k * m);

    time = 0:0.005:1; % Simulate 1 second of flight
    w_motor = 200; % Motor frequency (rad/s)
    F0 = 15; % 15N of unbalanced motor force
    F_t = F0 * sin(w_motor * time);

    % Define the Transfer Function
    sys = tf(1, [m c k]);

    % Run the simulation silently (no plotting)
    [x, ~] = lsim(sys, F_t, time);

    % Extract the maximum absolute deflection and convert to mm
    max_deflection = max(abs(x)) * 1000;

    % Save the result into our data table
    data.Max_Deflection_mm(i) = max_deflection;

    % Print progress update
    if mod(i, 10) == 0
        fprintf('Simulated %d / 100 booms...\n', i);
    end
end

toc; % Stop timer
disp('Saving fully labeled dataset...');
writetable(data, output_csv);
disp('✅ Labeling Complete! The dataset is ready.');