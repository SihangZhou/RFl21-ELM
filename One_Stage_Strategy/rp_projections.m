function W = rp_projections(D,d,kernel,Delta)
switch kernel
    case 'gaussian'
        W = sqrt(2)*normrnd(0,Delta,d,D);
    otherwise
        error('cannot sample from that yet.')
end
end
