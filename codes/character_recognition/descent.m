function [nnparam,J]= descent(Jin,nnparam,gn,n)
eta=0.002;
J=0;

for i=1:n
    [J,grad]=Jin(nnparam,gn);
    
    fprintf('\nIterations:%d||Cost:%f\n',i,J);
    nnparam=nnparam-eta*grad;
    gn=gn+1;
    if gn==3
        gn=0;
    end    
end    
