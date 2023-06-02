function plot_errores(erroresL2, erroresLinf)

plot(erroresL2)
hold on
plot(erroresLinf)
title('Errores con el cambio de la n')
legend('Error L_2','Error L_{\infty}');

end