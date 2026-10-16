% =========================================================================
% NOYRON 2.0 - PHASE 4: STRUCTURAL DYNAMICS
% SDOF Harmonic Excitation Analysis
% =========================================================================

% 1. IMPORT THE GEOMETRY
workspace_dir = 'D:\nyron 2\noyron_workspace';
filename = fullfile(workspace_dir, 'Boom_Standard.stl');

disp('Loading Noyron Geometry...');
boom_mesh = stlread(filename);

% Set up the dashboard
figure('Name', 'Noyron Structural Analysis', 'Color', 'w', 'Position', [100, 100, 800, 600]);

% Plot the STL
subplot(2,1,1);
trisurf(boom_mesh.ConnectivityList, boom_mesh.Points(:,1), ...
    boom_mesh.Points(:,2), boom_mesh.Points(:,3), ...
    'FaceColor', [0.3 0.3 0.3], 'EdgeColor', 'none');
camlight; lighting gouraud; material dull;
title('Imported Drone Boom (Standard Variant)');
axis equal; view(3); grid on;

% 2. SDOF SYSTEM PARAMETERS (Carbon Fiber Approximation)
disp('Calculating SDOF parameters...');
m = 0.45;           % Modal mass (kg)
k = 15000;          % Bending stiffness (N/m)
zeta = 0.02;        % Damping ratio (2% typical for carbon composites)

wn = sqrt(k/m);     % Natural frequency (rad/s)
fn = wn / (2*pi);   % Natural frequency (Hz)
c = 2 * zeta * sqrt(k*m); % Damping coefficient

fprintf('Natural Frequency: %.2f Hz\n', fn);

% 3. HARMONIC EXCITATION (Motor unbalance simulation)
disp('Running Vibration Simulation...');
t = 0:0.001:2;               % 2 seconds of flight time
w_motor = 200;               % Motor operating frequency (rad/s)
F0 = 15;                     % 15N of force from rotor unbalance
F_t = F0 * sin(w_motor * t); % Harmonic force profile

% Define the Transfer Function: G(s) = 1 / (ms^2 + cs + k)
sys = tf(1, [m c k]);

% Simulate the response using lsim
[x, t_out] = lsim(sys, F_t, t);

% 4. PLOT VIBRATION RESPONSE
subplot(2,1,2);
plot(t_out, x * 1000, 'b', 'LineWidth', 1.5); % Convert meters to mm for readability
title(sprintf('Boom Tip Vibration Response (Motor w = %d rad/s)', w_motor));
xlabel('Time (seconds)');
ylabel('Deflection (mm)');
grid on;