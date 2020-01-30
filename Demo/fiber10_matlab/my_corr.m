function out=my_corr(img1,img2)


m1=mean(mean(img1,1),2);
m2=mean(mean(img2,1),2);

sig1_2=mean(mean(((img1-repmat(m1,size(img1,1),size(img1,2))).*(img2-repmat(m2,size(img2,1),size(img2,2)))),1),2);
sig1=mean(mean(((img1-repmat(m1,size(img1,1),size(img1,2))).*(img1-repmat(m1,size(img1,1),size(img1,2)))),1),2);
sig2=mean(mean(((img2-repmat(m2,size(img2,1),size(img2,2))).*(img2-repmat(m2,size(img2,1),size(img2,2)))),1),2);
% t1=2*m1*m2+(0.01*L)^2;
% t2=2*sig1_2+(0.03*L)^2;
% t3=m1^2+m2^2+(0.01*L)^2;
% t4=sig1+sig2+(0.03*L)^2;


out=sig1_2./sqrt(sig1.*sig2);



end






