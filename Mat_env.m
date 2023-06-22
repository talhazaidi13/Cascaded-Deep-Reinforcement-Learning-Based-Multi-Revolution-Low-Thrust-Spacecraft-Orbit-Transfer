
classdef Mat_env < handle
    
    properties (SetAccess = private)
    
    %%%%%%% Computing inputs for the testcase
    mu=0; % Gravitational parameter
    a=0; % Semi-major axis of initial orbit in km
    e0=0; % eccentricity of initial orbit
    inc =0;      %Inclination of initial orbit in Deg
    rp=0; % Perigee radius of initial orbit
    ra=0;% Apogee radius of initial orbit
    vp=0; % Velocity at the perigee of initial orbit
    rc =[]; % Position vector of spacecraft at perigee of the initial orbit in the body fixed frame (3-1-3 rotation)
    v=[]; %Velocity vector of spacecraft at perigee of the initial orbit (3-1-3 rotation)
    RRR=[]; % Rotation matrix considering RAAN=0 deg and argument of latitude=0
    r=[]; % Position vector of spacecraft at perigee of the initial orbit in the Inertial frame
    rv = [];
    Param =[];
    h0 =0;
    hx0 = 0;
    hy0 = 0;
    ex0 = 0;
    ey0 = 0;
    state0 = []; % Converting the inertial position and velocity vectors into the suwat's dynamical coordinates
    state = [];  %[h;hx;hy;ex;ey;phi;time] where m=m0-time*m_dot
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %The above computution is just required once to convert initial states into
    %dynamic coordinates
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    h_limit=0;
    F=0; %%0.3115/1000; %in kilo Newtons
    I_sp = 0; % in sec
    m0=0; % kg
    alpha=0; % in radians
    beta=0;% in radians
    segment=0; %  segment length (angle) after which the end states are compouted in radians
    
    % non-dimensional values for reference
    DU = 0;                 %distance unit, Km
    TU = 0;  %time unit, s
    SU = 0;                 %speed unit, Km/sec
    MU =0;                    %mass Unit, Kg
    HU=0;  % angular momentum
    FU = 0;       %force unit(K-N)
    
    g0=0; % m/sec^2
    m_dot = 0; % in kg/sec
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % %Terminal desired state values for reference
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % h(end)=129640.2292 %km^2/sec
    % hx(end)=0;
    % hy(end)=0;
    % ex(end)=0;
    % ey(end)=0;
    end
    
    methods
        function obj = Mat_env ()      % constructor
        
        obj.mu=398600.4418; % Gravitational parameter
        a=41145.4922; % Semi-major axis of initial orbit in km  # maxGEO=42164   initialGTO=24364 /// 40000 
        e0=0.0071; % eccentricity of initial orbit       # maxGEO=0  initialGTO=0.7306  //0.2
        inc=4.9085;      %Inclination of initial orbit in Deg        # # maxGEO=0  initialGTO=28.5  //10
        obj.ra=a*(1+e0);
        obj.rp=a*(1-e0);
        a = (obj.rp+obj.ra)/2;
        ex0 = (obj.ra/a-1);
        ey0 = 0;
        h0 = obj.mu*sqrt(a*(1-ex0^2));   
        hx0 = -sin(inc/180*pi)*h0;
        hy0 = 0;
        m0 = 1;
                 
        state =[h0;hx0;hy0;ex0;ey0;0;0;0];  %[h;hx;hy;ex;ey;phi;time;fuel burnt] 
        %[h;hx;hy;ex;ey;phi;time] where m=m0-time*m_dot
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %The above computution is just required once to convert initial states into
        %dynamic coordinates
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        format long
        obj.h_limit= 42164*sqrt(obj.mu/(42164));
        obj.F=1.17/1000;%0.3115/1000; %in kilo Newtons
        obj.I_sp = 1800; % in sec
        obj.m0=2000; % kg
        obj.alpha=0.5; % in radians
        obj.beta=0.5;% in radians
        obj.segment=10*pi/180; %  segment length (angle) after which the end states are compouted in radians
        
        %obj.g0=9.81; % m/sec^2
        %obj.m_dot = -obj.F/obj.I_sp/obj.g0; % in kg/sec
        
        % non-dimensional values for reference
        obj.DU = 42164;                 %distance unit, Km
        obj.TU = sqrt(42164^3/398600);  %time unit, s
        obj.SU = obj.DU/obj.TU;                 %speed unit, Km/sec
        obj.MU = 5000;                    %mass Unit, Kg
        obj.HU=obj.DU*obj.SU;  % angular momentum
        obj.FU = obj.MU*obj.DU*1000/obj.TU^2;       %force unit(K-N)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % %Terminal desired state values for reference
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % h(end)=129640.2292 %km^2/sec
        % hx(end)=0;
        % hy(end)=0;
        % ex(end)=0;
        % ey(end)=0;

        %write = [obj.state(1), obj.state(2), obj.state(3), obj.state(4), obj.state(5), obj.state(6), obj.state(7), obj.alpha,obj.beta,obj.F,obj.segment]
        %csvwrite("csvlist.dat", write)
        end

        
    end

    
    
   methods(Static)
       function result = resulting()
        %global mu % defined in chkStop
        mu = 398600.4418;
        M = csvread('E:/RL_project_outputs/Training_testing_code/csv_files/csvlist.dat');
        state = M(1:8)';
        alpha = M(9);
        beta = M(10);
        F = M(11);
        segment = M(12);
        m0 = M(13);
        I_sp=M(14);
        %display(state)
        [finalState, finalSpacecraftMass]=spacecraftEnivironment(state,alpha,beta,F,segment,m0,I_sp); % testing the environment function
        PropellentBurnt=abs(finalState(1,8));
        finalState1=finalState;
        finalSpacecraftMass1=finalSpacecraftMass;
        
        % non-dimensional values for reference
        DU = 42164;                 %distance unit, Km
        TU = sqrt(42164^3/398600);  %time unit, s
        SU = DU/TU;                 %speed unit, Km/sec
        MU = 5000;                    %mass Unit, Kg
        HU=DU*SU;  % angular momentum
        FU = MU*DU*1000/TU^2;       %force unit(K-N)
        e0=0.7306;
        
        p =finalState(1,1)^2/mu;
        e = sqrt(finalState(1,4)^2+finalState(1,5)^2);
        a=p/(1-e^2);
        i=( (asin(sqrt(finalState(1,2)^2+finalState(1,3)^2)/finalState(1,1)) )/pi)*180;
        %i=(asin(sqrt(finalState(1,2)^2+finalState(1,3)^2)/finalState(1,1))) * (pi/180);

        %flag=chkStop( finalState(1,1)/HU,finalState(1,2)/HU,finalState(1,3)/HU,finalState(1,4),finalState(1,5));
        flag=chkStop( finalState(1,1),finalState(1,2),finalState(1,3),finalState(1,4),finalState(1,5));

        if flag==1
            %disp('Matlab 236 :Terminal conditions reached')
%             break
        end
        if  i >=89.9
            disp('Matlab 240 :Inclination is aprroching 90 deg')
%             break
        end
        if a>= 42164*2
            disp('Matlab 244 : Energy above threshold')
%             break
        end
        if e> e0+(e0*0.2)
            disp('Matlab 248 :Eccentricity above threshold')
%             break
        end
              
        result= [finalState1, finalSpacecraftMass1];

       end
     end

  methods(Static)
       function eclipse_flag = chkEclipse()
            %global mu % defined in chkStop
            mu = 398600.4418;
            M = csvread('E:/RL_project_outputs/Training_testing_code/csv_files/csvlist.dat');
            state = M(1:8)';
            alpha = M(9);
            beta = M(10);
            F = M(11);
            segment = M(12);
            m0 = M(13);
            I_sp=M(14);
            %display(state)
            rEarth=6378.1363;
    
            mu=398600.4415;
            h = M(1);
            hx = M(2);
            hy = M(3);
            ex = M(4);
            ey = M(5);
            phi= M(6);
            
            hz = sqrt(h^2-hx^2-hy^2);
            r = h^2/mu/(1+ex*cos(phi)+ey*sin(phi));
            r = [r; zeros(2,1)];
            
            % Ro_dash2G( H )
            % =============== Error ===================================================% ===============% ===============
            H = [hx;hy;hz];
            [m,n] = size( H );
            if (m~=3)||(n~=1)
                error('Input has to be 3x1 vector ([hx;hy;hz])');
            end
            % ================ Code ===================================================% ===============% ===============% ===============
            hx_1 = H(1);
            hy_1 = H(2);
            hz_1 = H(3);
            
            x_dash_hat = [hz_1;0;-hx_1]/sqrt(hx_1^2+hz_1^2);
            h_hat = [hx_1;hy_1;hz_1]/sqrt(hx_1^2+hy_1^2+hz_1^2);
            xn_dash_hat = [0         -h_hat(3) h_hat(2);
                           h_hat(3)     0       -h_hat(1);
                          -h_hat(2)   h_hat(1)    0]*x_dash_hat;
            Rotation = [x_dash_hat xn_dash_hat h_hat];
    
            % ===============% ===============% ===============% ===============% ===============% ===============
            %  Ro_rnh2dash(theta)
            % =============== Error ===================================================
            if ~isscalar(phi)
                error('theta has to be scalar')
            end
            % ================ Code ===================================================
            Rotation_2 = [cos(phi) -sin(phi) 0;
                        sin(phi)  cos(phi) 0;
                        0 0 1];
            % ===============% ===============% ===============% ===============% ===============% ===============

            Rxyz = Rotation* Rotation_2 *r;
            if Rxyz(1)>0 && sqrt(Rxyz(2)^2 + Rxyz(3)^2)<rEarth
                isEcl = 1;
            else
                isEcl = 0;   
            end

            eclipse_flag = isEcl;
       end
  end

    methods(Static)
       function eclipse_flag_new = chkEclipse_new()
            %global mu % defined in chkStop
            mu = 398600.4418;
            M = csvread('E:/RL_project_outputs/csvlist.dat');
            state = M(1:8)';
            alpha = M(9);
            beta = M(10);
            F = M(11);
            segment = M(12);
            m0 = M(13);
            I_sp=M(14);
            %display(state)
            rEarth=6378.1363;
    
            mu=398600.4415;
            h = M(1);
            hx = M(2);
            hy = M(3);
            ex = M(4);
            ey = M(5);
            phi= M(6);
            
            hz = sqrt(h^2-hx^2-hy^2);
            r = h^2/mu/(1+ex*cos(phi)+ey*sin(phi));
            r = [r; zeros(2,1)];
            
            % Ro_dash2G( H )
            % =============== Error ===================================================% ===============% ===============
            H = [hx;hy;hz];
            [m,n] = size( H );
            if (m~=3)||(n~=1)
                error('Input has to be 3x1 vector ([hx;hy;hz])');
            end
            % ================ Code ===================================================% ===============% ===============% ===============
            hx_1 = H(1);
            hy_1 = H(2);
            hz_1 = H(3);
            
            x_dash_hat = [hz_1;0;-hx_1]/sqrt(hx_1^2+hz_1^2);
            h_hat = [hx_1;hy_1;hz_1]/sqrt(hx_1^2+hy_1^2+hz_1^2);
            xn_dash_hat = [0         -h_hat(3) h_hat(2);
                           h_hat(3)     0       -h_hat(1);
                          -h_hat(2)   h_hat(1)    0]*x_dash_hat;
            Rotation = [x_dash_hat xn_dash_hat h_hat];
    
            % ===============% ===============% ===============% ===============% ===============% ===============
            %  Ro_rnh2dash(theta)
            % =============== Error ===================================================
            if ~isscalar(phi)
                error('theta has to be scalar')
            end
            % ================ Code ===================================================
            Rotation_2 = [cos(phi) -sin(phi) 0;
                        sin(phi)  cos(phi) 0;
                        0 0 1];
            % ===============% ===============% ===============% ===============% ===============% ===============

            Rxyz = Rotation* Rotation_2 *r;
            if Rxyz(1)<0 && sqrt(Rxyz(2)^2 + Rxyz(3)^2)<rEarth
                isEcl = 1;
            else
                isEcl = 0;   
            end

            eclipse_flag_new = isEcl;
       end
  end


end
