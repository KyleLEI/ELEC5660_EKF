addpath('./utils');
syms x y z % PNP: position
syms roll pitch yaw % PNP: orientation
syms vx vy vz % OPTFLOW: linear velocity
syms bgx bgy bgz % gyro bias
syms bax bay baz % acc bias
syms wx wy wz % SENSOR: gyro measurement
syms ax ay az % SENSOR: acc measurement
g = sym([0 0 -9.81]'); % gravity
syms ngx ngy ngz % gyro noise
syms nax nay naz % acc noise
syms nbgx nbgy nbgz % gyro bias noise
syms nbax nbay nbaz % acc bias noise
syms a b c % rubbish vars for jacobian computation
assume([x y z roll pitch yaw vx vy vz bgx bgy bgz bax bay baz],'real');
assume([ngx ngy ngz nax nay naz],'real');
assume([wx wy wz ax ay az g nbgx nbgy nbgz nbax nbay nbaz],'real');

p = [x y z]'; % x1
q = [roll pitch yaw]'; % x2
p_dot = [vx vy  vz]'; % x3
bg = [bgx bgy bgz]'; % x4
ba = [bax bay baz]'; % x5

% x
X = [p;q;p_dot;bg;ba];
%u
wm = [wx wy wz]';
am = [ax ay az]';
u = sym([zeros(3,1);wm;am;zeros(6,1)]);
% n
ng = [ngx ngy ngz]';
na = [nax nay naz]';
nbg = [nbgx nbgy nbgz]';
nba = [nbax nbay nbaz]';
n = [a;b;c;ng;na;nbg;nba];

R = RPYtoRot_ZXY(roll,pitch,yaw)'; % R_WB = R_BW'
G = [cos(pitch) 0 -cos(roll)*sin(pitch);
     0 1 sin(roll);
     sin(pitch) 0 cos(roll)*cos(pitch)]; % G_BW, related to x2(q)
G_inv = simplify(inv(G)); % ~G_WB
X_dot = [p_dot;G_inv*(wm-bg-ng);g+R*(am-ba-na);nbg;nba];
f = simplify(X_dot);
f0 = [p_dot;G_inv*(wm-bg);g+R*(am-ba);zeros(6,1)];

% Calculate A,U
A = jacobian(f0,X);
U = jacobian(f,n);

% Calculate C
Z = [p;q;R'*p_dot];
C = jacobian(Z,X);
g0 = [p;q;R'*p_dot];