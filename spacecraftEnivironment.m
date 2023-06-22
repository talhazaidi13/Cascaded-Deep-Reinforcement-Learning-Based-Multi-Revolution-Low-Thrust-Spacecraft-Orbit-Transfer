function[finalState, finalSpacecraftMass]=spacecraftEnivironment(state,alpha,beta,F,segment,m0,Isp)
% segment = angle in radians 
% F= thrust in kilo newtons
% alpha, beta = thrust angles in radians
%display("line 5: ");
%display(state);
n=10; % integration steps
phi_step=segment/n;
phi_span = state(6):phi_step:n*phi_step+state(6);
 ctrl_res=[alpha*ones(length(phi_span)-1,1);beta*ones(length(phi_span)-1,1)];
    [~,tra_d] = ode45(@(phi,x)TwoBody(phi,x,ctrl_res,n,state(6),F,phi_step,m0,Isp)...
        ,phi_span,state);
finalState=tra_d(end,:);
finalSpacecraftMass=m0+tra_d(end,8);
end

function dst_dphi = TwoBody(phi,state,u,n,phi0,Thr,phi_step,m0,Isp)
%global mu     % defined in chkStop
mu = 398600.4418;
g0=9.81/1000; % km/sec^2
if (phi-phi0) > 0
    if (phi-phi0) >= n*phi_step
        alp = u(n);
        bet = u(n*2);
    else
        alp = u(ceil((phi-phi0)/phi_step));
        bet = u(n + ceil((phi-phi0)/phi_step));
    end
    
else
    alp = u(1);
    bet = u(n+1);

%display("line 33: ");
%display(state);

end

[G,f] = Var_mat([state(1:5);phi],mu);

h = state(1);
hx = state(2);
hy = state(3);
ex = state(4);
ey = state(5);
m = m0 +state(8);
%display("line 45: ");
%display(state);

%display("line 46: ");
%display(state(1));
%display(state(2));
%display(state(3));



if h < hy
    display('h less than hy')
    display(h)
    display(hx)
    display(hy)
    error('xxx')
end

B = 1+ex*cos(phi)+ey*sin(phi);
fr = -sin(alp).*cos(bet);
fn =  cos(alp).*cos(bet);
fh =  sin(bet);
F = Thr*[fr; fn; fh];
U = F + [0;0;0];   

dt_dphi = h^3*m*mu*B*sqrt(h^2-hy^2)/...
        (m*mu^3*B^3*sqrt(h^2-hy^2)-h^4*hy*sin(phi)*U(3));


dOrb_dphi = (f(1:5,1) + 1/m*G(1:5,:)*U)*dt_dphi;
dm_dphi=(-Thr/g0/Isp)*dt_dphi;
dst_dphi = [dOrb_dphi;1;dt_dphi;dm_dphi];
end

function  [G,f]  = Var_mat( state,mu )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
h   = state(1);
hx  = state(2);
hy  = state(3);
ex  = state(4);
ey  = state(5);
phi = state(6);

A = ex*sin(phi) - ey*cos(phi);
B = 1 + ex*cos(phi) + ey*sin(phi);

g12 = h^2/mu/B;
g22 = h*hx/mu/B;
g23 = 1/mu/B/sqrt(h^2-hy^2)*(h^2*sqrt(h^2-hx^2-hy^2)*sin(phi) + ...
      h*hx*hy*cos(phi));                        %correct
g32 = h*hy/mu/B;                                %correct
g33 = -h*sqrt(h^2-hy^2)/mu/B*cos(phi);          %correct
g41 = h*sin(phi)/mu;                            %correct
g42 = 2*h*cos(phi)/mu + h*(A/mu/B)*sin(phi);      %correct
g43 = h*ey*hy/mu/B/sqrt(h^2-hy^2)*sin(phi);     %correct
g51 = -h/mu*cos(phi);                           %correct
g52 = 2*h/mu*sin(phi) - h*A/mu/B*cos(phi);      %correct
g53 = -h*ex*hy/mu/B/sqrt(h^2-hy^2)*sin(phi);    %correct
g63 = -h*hy/mu/B/sqrt(h^2-hy^2)*sin(phi);

G = [0 g12 0; 0 g22 g23; 0 g32 g33; g41 g42 g43; g51 g52 g53; 0 0 g63];
f = [0;0;0;0;0; mu^2*B^2/h^3];
end


