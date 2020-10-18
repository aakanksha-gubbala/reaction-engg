
function [] = CSTR_Reactions()
    function y = CSTR_solver(t, x0)
        function f = F(x)
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

                f(1) = MEA + (r2 - r1)*t;
                f(2) = DEA + (r3 - r2)*t;
                f(3) = TEA + (-r3)*t;
        end
    y = fsolve(@F, x0);
    end

    t = linspace(0, 15, 100);
    n = length(t);
    m = zeros(n);
    d = zeros(n);
    te = zeros(n);
    for q = 1:n
        y = CSTR_solver(t(q), [0 0 0]);
        m(q) = y(1);
        d(q) = y(2);
        te(q) = y(3);
    end
    plot(t, [m, d, te], 'linewidth', 2)
    ylim([0 1])
    xlabel("Space time in min")
    ylabel("Concentration in M")
end


