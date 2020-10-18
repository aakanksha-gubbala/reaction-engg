
function [] = parallelReactions(t, x)
    function [f] = dXdT(t, x)
        A = x(1);
        V = x(2);
        D = x(3);

        k1 = 1;
        k2 = 0.4;

        dAdt = -k1*A - 2*k2*A*A;
        dVdt = k1*A;
        dDdt = k2*A*A;
        f = [dAdt; dVdt; dDdt]; 
    end
[t,y] = ode15s(@dXdT,t, x);
figure
plot(t, y,'linewidth', 2)
legend({'[A]', '[V]', '[D]'},'Location','bestoutside')
legend('boxoff')
xlabel("t in min")
ylabel("Concentration in M")
end