function i=RouletteWheelSelectionM(P)

    r=rand;
    
    rho=2.5;
    C=exp(-(1/rho)*P);
    i=find(C == max(C(:)));
end