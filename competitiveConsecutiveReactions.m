
function [] = competitiveConsecutiveReactions(t, x)
    function [f] = dXdT(t, x)
        MEA = x(1);
        DEA = x(2);
        TEA = x(3);

        k1 = 1;
        k2 = 0.4;
        k3 = 0.1;
        
        A0 = 1;
        EO0 = 2.4;
        
        A = A0 - MEA - DEA - TEA;
        EO = EO0 - MEA - 2*DEA - 3*TEA;
        
        r1 = k1*A*EO;
        r2 = k2*MEA*EO;
        r3 = k3*DEA*EO;

        dMEAdt = r1 - r2;
        dDEAdt = r2 - r3;
        dTEAdt = r3;
        f = [dMEAdt; dDEAdt; dTEAdt]; 
    end
[t,y] = ode15s(@dXdT,t, x);
figure
plot(t, y,'linewidth', 2)
legend({'[MEA]', '[DEA]', '[TEA]'},'Location','bestoutside')
legend('boxoff')
ylim([0 1])
xlabel("Space time in min")
ylabel("Concentration in M")
end