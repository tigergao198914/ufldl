function weight = getweight( W, padw, padh, fw, fh )
    wnum = (size(W,1)-fw)/padw+1;
    hnum = (size(W,2)-fh)/padh+1;
    weight = zeros(fw*fh, wnum*hnum);
    for i=1:hnum
        for j=1:wnum 
            t = W( 1+padw*(j-1):1+padw*(j-1)+fw-1, 1+padh*(i-1):1+padh*(i-1)+fh-1);
            weight(:, (i-1)*wnum+j ) = reshape(t, fw*fh, 1 );
        end
    end
end
