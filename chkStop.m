function flag = ...
    chkStop( h,hx,hy,ex,ey)
global mu
mu= 398600.4418;
p = h^2/mu;

tol_inc = 0.01;                  % tolerance of inclination +- deg
tol_ecc = 0.0001;             % 0.00001 tolerance of eccentricity +0
%tol_a   = 0.00001*35786;             % tolerance of normalize a +- DU
%tol_a   = 35786;             % tolerance of normalize a +- DU
target_a   = 42164;             % tolerance of normalize a +- DU
tol_a = target_a * (0.00001);
%==============================================
ecc = sqrt(ex^2+ey^2);
if ecc < (tol_ecc)
    flag_ecc = 1;
else
    flag_ecc = 0;
end
%==============================================
a = p/(1-ecc^2);
%if (target_a -tol_a) < a && a < (target_a + (2*tol_a))
if (target_a) < a && a < (target_a + (2*tol_a))
    flag_a = 1;
else
    flag_a = 0;
end
%==============================================
i = ((asin(sqrt(hx^2+hy^2)/h))/pi)*180;
%i = (asin(sqrt(hx^2+hy^2)/h))* (pi/180);

if i < tol_inc
    flag_inc = 1;
else
    flag_inc = 0;
end
%==============================================


if flag_a && flag_ecc && flag_inc
    flag = 1;
else
    flag = 0;
end

end

