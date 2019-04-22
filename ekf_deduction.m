addpath('./utils');
syms x y z % position
syms roll pitch yaw % orientation
syms vx vy vz % linear velocity
syms bgx bgy bgz % gyro bias
syms bax bay baz % acc bias
syms wx wy wz % gyro measurement
syms ax ay az % acc measurement
g = [0 0 -9.81]'; % gravity
syms ngx ngy ngz % gyro noise
syms nax nay naz % acc noise
syms nbgx nbgy nbgz % gyro bias noise
syms nbax nbay nbaz % acc bias noise

assume([x y z roll pitch yaw vx vy vz bgx bgy bgz bax bay baz],'real');
assume([ngx ngy ngz nax nay naz],'real');
assume([wx wy wz ax ay az g nbgx nbgy nbgz nbax nbay nbaz],'real');
p = [x y z]';
q = [roll pitch yaw]';
p_dot = [vx vy  vz]';
bg = [bgx bgy bgz]';
ba = [bax bay baz]';
X = [p;q;p_dot;bg;ba];
wm = [wx wy wz]';
am = [ax ay az]';
ng = [ngx ngy ngz]';
na = [nax nay naz]';
nbg = [nbgx nbgy nbgz]';
nba = [nbax nbay nbaz]';

R = RPYtoRot_ZXY(roll,pitch,yaw)'; % R_WB
G = [cos(pitch) 0 -cos(roll)*sin(pitch);
     0 1 sin(roll);
     sin(pitch) 0 cos(roll)*cos(pitch)]; % G_BW
G_inv = simplify(inv(G)); % G_WB
X_dot = [p_dot;G_inv*(wm-bg-ng);g+R*(am-ba-na);nbg;nba];
disp(simplify(X_dot));