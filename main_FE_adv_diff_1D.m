% Testing the FE assembly code for advection-diffusion problems

%% First test case: u(x) = sin(2*pi*x)

a = 1;
b = 1;

n = 10;

% exact solution for test case
u = cell(1,1);
u{1,1} = @(x) sin(2*pi*x);

% RHS of -au''(x) + bu'(x) = f(x) based on exact solution u
f = cell(1,1);
f{1,1} = @(x) 4*pi*pi*sin(2*pi*x) + 2*pi*cos(2*pi*x);

% creating the x grid
xgrid = [0:1/n:1]';
x0 = xgrid(1);
x1 = xgrid(end);

% spatial discretization
[A_bc,M_bc,F,~] = FE_adv_diff_1D(n,a,b,f,x0,x1);

% finding a solution in the FE space
U = A_bc\F;

% inserting correct BCs
U_vis = [0;U;0];

U_exact = u{1,1}(xgrid);

% plotting the numerical solution
figure(1)
p1 = plot(xgrid,U_vis);
p1.LineStyle = "--";
p1.Marker = '.';
p1.LineWidth = 1;
p1.MarkerSize = 10;
p1.Color = 'r';

hold on

% plotting the exact solution
p2 = plot(xgrid,U_exact);
p2.LineStyle = "-";
p2.Color = 'b';
p2.LineWidth = 1;

hold on
legend([p1 p2],'Numerical Solution','Exact Solution');

%% Second test case: u(x) = x(1-x)/2 with u(0) = u(1) = 0

% we keep the same a, b, n values

% exact solution and RHS:
u1 = cell(1,1);
u1{1,1} = @(x) x.*(1-x)./2;

f1 = cell(1,1);
f1{1,1} = @(x) (3/2) - x;

% FE implementation, solving for coefficients, and adding in BCs
[A1,M1,F1,~] = FE_adv_diff_1D(n,a,b,f1,x0,x1);

U1 = A1\F1;

U1_vis = [0;U1;0];

U1_exact = u1{1,1}(xgrid);

% plotting the numerical solution
figure(2)
q1 = plot(xgrid,U1_vis);
q1.LineStyle = "--";
q1.Marker = '.';
q1.LineWidth = 1;
q1.MarkerSize = 10;
q1.Color = 'r';

hold on

% plotting the exact solution
q2 = plot(xgrid,U1_exact);
q2.LineStyle = "-";
q2.Color = 'b';
q2.LineWidth = 1;

hold on
legend([q1 q2],'Numerical Solution','Exact Solution');

%% Error analysis

N = [50,100,200,400];
h = 1./N;
error_norms = zeros(1,4);

tolquad = 1e-12;
du_dx = @(x) 2*pi*cos(2*pi*x);
first_int = 2*pi*pi;

for i = 1:4
    mesh = [0:h(i):1]';
    [Aerr,~,Ferr,~] = FE_adv_diff_1D(N(i),a,b,f,x0,x1);
    Uerr = Aerr\Ferr;
    U_ex = u{1,1}(mesh);
    U_ex = U_ex(2:end-1);
    % errvec = abs(U_ex - Uerr);
    % error_norms(i) = norm(sqrt(1/N(i)).*errvec); % computes the error in the 2-norm with the important sqrt(h) scaling factor
    % now we compute the Sobolev norm:
    error_norms(i) = sobolev_norm(first_int,h(i),Uerr,Aerr,du_dx,x0,tolquad);
end

% producing a best-fit line for the data
h = 1./N;
logh = log(h);
logE = log(error_norms);
X = [ones(1,4); logh];
betas = X'\logE';
logC = betas(1);
kappa = betas(2);

% plotting

figure(3)
p5 = plot(logh,logE);
p5.LineStyle = "none";
p5.Marker = '.';
p5.Color = 'r';
p5.LineWidth = 1;
p5.MarkerSize = 12;

hold on
p6 = plot(logh,logC+kappa.*logh);
p6.LineStyle = "--";
p6.Color = 'b';
p6.LineWidth = 1;

grid on

hold on
legend([p5 p6],'Error Norm','Best-fit Line');
xlabel('$\log(h)$','Interpreter','latex');
ylabel('$\log(||u - u_h||_{H^1(D)})$','Interpreter','latex');

%% Calculating the error in the Sobolev norm

% for this section we are looking to compute |int[(u' - (u_h)')^2]|

% we have that the above integral is equal to
% int((u')^2) + int((u_h)'^2) - 2*int((u_h)'*u')