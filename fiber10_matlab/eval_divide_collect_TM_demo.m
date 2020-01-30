close all
tic
load('transmit')
delete('train_dataF.bin')
delete('train_labelsF.bin')
delete('train_dataF_1.bin')
delete('train_labelsF_1.bin')


ff2OUT=0;
TEST=0;
flag_offset_eval_examples=0;


NUMEVAL_SAMPLES=1000;

%% The remaining parameters 
NUMEVAL_SAMPLES2=1000;
NUM=NUMEVAL_SAMPLES2;
HEIGHT=51;
WIDTH=51;
dim=200;
HEIGHT2=dim;
WIDTH2=dim;
len2=dim-1;
intensity=0;
%% hooks
COR_avg=[];
PSNR_avg=[];
MSE_avg=[];
COR_avg_FOV=[];
COR_avg_intensity=[];
COR_avg_intensity_FOV=[];
COR_avg_G=[];
COR_avg_intensity_G=[];

%% body of the code starts here

n1=floor(NUMEVAL_SAMPLES/NUMEVAL_SAMPLES2);
res=NUMEVAL_SAMPLES-n1*NUMEVAL_SAMPLES2;
B=zeros(HEIGHT,WIDTH,NUMEVAL_SAMPLES2);
B_sin=zeros(HEIGHT,WIDTH,NUMEVAL_SAMPLES2);


D=zeros(HEIGHT2,WIDTH2,NUMEVAL_SAMPLES2);
G=zeros(HEIGHT2,WIDTH2,NUMEVAL_SAMPLES2);
G_sin=zeros(HEIGHT2,WIDTH2,NUMEVAL_SAMPLES2);
for i=1:n1
    str=['Output_test_phases_' num2str(i) '.bin'];
    str_1=['Output_test_phases_sin_' num2str(i) '.bin'];
    str3=['eval_dataF_' num2str(i) '.bin'];
    str4=['Output_test_phases_G_' num2str(i) '.bin'];
    str4_1=['Output_test_phases_G_sin_' num2str(i) '.bin'];
    V=(1:NUMEVAL_SAMPLES2);
    V2=(1:NUMEVAL_SAMPLES2);
    

    
    f=fopen(str,'r'); 
    A=fread(f,HEIGHT*WIDTH*NUMEVAL_SAMPLES2,'uint8');
    B(:,:,V)=reshape(A,[HEIGHT,WIDTH,NUMEVAL_SAMPLES2]);
    fclose(f);
    
    
    f=fopen(str_1,'r'); 
    A_1=fread(f,HEIGHT*WIDTH*NUMEVAL_SAMPLES2,'uint8');
    B_sin(:,:,V)=reshape(A_1,[HEIGHT,WIDTH,NUMEVAL_SAMPLES2]);
    fclose(f);
   
    
    gg=fopen(str3,'r'); 
    A2=fread(gg,HEIGHT2*WIDTH2*NUMEVAL_SAMPLES2,'uint8');
    D(:,:,V2)=reshape(A2,[HEIGHT2,WIDTH2,NUMEVAL_SAMPLES2]);
    fclose(gg);
    
    gg=fopen(str4,'r'); 
    A2=fread(gg,HEIGHT2*WIDTH2*NUMEVAL_SAMPLES2,'uint8');
    G(:,:,V2)=reshape(A2,[HEIGHT2,WIDTH2,NUMEVAL_SAMPLES2]);
    fclose(gg);

    gg=fopen(str4_1,'r'); 
    A2_1=fread(gg,HEIGHT2*WIDTH2*NUMEVAL_SAMPLES2,'uint8');
    G_sin(:,:,V2)=reshape(A2_1,[HEIGHT2,WIDTH2,NUMEVAL_SAMPLES2]);
    fclose(gg);

B=B/255;
% B=2*B-1;
B=double(B);
B_sin=B_sin/255;
% B_sin=2*B_sin-1;
B_sin=double(B_sin);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[ff1,~]=clip_mask(holo_params.freq.maskc);
[ff2,~]=clip_mask(slm_params.freq.maskc);

if ff2OUT==1
output=zeros(size(ff1,1),size(ff1,2),NUMEVAL_SAMPLES2);
len1=size(ff1,1);
else
  output=zeros(size(holo_params.freq.maskc,1),size(holo_params.freq.maskc,2),NUMEVAL_SAMPLES2);
len1=size(holo_params.freq.maskc,1);  
end
v=(floor((len1-len2)/2)):(floor((len1+len2)/2));


for o=1:NUMEVAL_SAMPLES2
    o;

 X2_exp=B(:,:,o)+1j*B_sin(:,:,o);
 X4=fftshift2(fft2(X2_exp));
 X5=mask(X4,ff2);
    

X5_1=T*X5;
if ff2OUT==1
X5_2=unmask(X5_1,ff1);
else  
X5_2=unmask(X5_1,holo_params.freq.maskc); 
end
X5_3=ifft2(ifftshift2(X5_2));
output(:,:,o)=X5_3;
end


% B=(B+1)/2;
B=B*255;
% B_sin=(B_sin+1)/2;
B_sin=B_sin*255;

if intensity==1
    OUT2=((abs(output(v,v,:))).^2)*255./max(max(((abs(output(v,v,:))).^2),[],1),[],2);
else
     OUT2=((abs(output(v,v,:))).^1)*255./max(max(((abs(output(v,v,:))).^1),[],1),[],2);
end


OUT3=zeros(size(OUT2,1),size(OUT2,2),NUM);
for ss=1:NUM  
    OUT3(:,:,ss)=(imadjust(OUT2(:,:,ss)/255,[100/255,255/255]))*255;  
end



    output=((abs(output(:,:,:))).^1)*255./max(max(((abs(output(:,:,:))).^1),[],1),[],2);
    if mod(abs(len1-HEIGHT2),2)==0
        DD=padarray(D,[floor((len1-HEIGHT2)/2), floor((len1-HEIGHT2)/2)],0,'both');
    else
        DD=padarray(D,[floor((len1-HEIGHT2)/2), floor((len1-HEIGHT2)/2)],0,'both');
        DD=padarray(DD,[1 1],0,'pre');
    end
    


    cor2=squeeze(my_corr(D(:,:,1:NUMEVAL_SAMPLES2),OUT2(:,:,1:NUMEVAL_SAMPLES2)));
    cor2_intensity=squeeze(my_corr(D(:,:,1:NUMEVAL_SAMPLES2),OUT2(:,:,1:NUMEVAL_SAMPLES2).^2));
    
    cor2_FOV=squeeze(my_corr(DD(:,:,1:NUMEVAL_SAMPLES2),output(:,:,1:NUMEVAL_SAMPLES2)));
    cor2_intensity_FOV=squeeze(my_corr(DD(:,:,1:NUMEVAL_SAMPLES2),output(:,:,1:NUMEVAL_SAMPLES2).^2));
    
    
    cor2_G=squeeze(my_corr(D(:,:,1:NUMEVAL_SAMPLES2),G(:,:,1:NUMEVAL_SAMPLES2)));
    cor2_intensity_G=squeeze(my_corr(D(:,:,1:NUMEVAL_SAMPLES2),G(:,:,1:NUMEVAL_SAMPLES2).^2)); 

%  
if i<3
cc=clock;
eee=double(OUT2(:,:,2).^2);
eee=uint8(255*eee/max(eee(:)));
imwrite(eee,['Output_' num2str(cc(4:5)) '.png']);
pause(1)


cc=clock;
eee=double(D(:,:,2));
eee=uint8(255*eee/max(eee(:)));
imwrite(eee,['Target_' num2str(cc(4:5)) '.png']);
pause(1)

cc=clock;
eee=B(:,:,2);
eee=uint8(255*eee/max(eee(:)));
imwrite(eee,['SLM_2D_input' num2str(cc(4:5)) '.png']);
pause(1)

end

    COR_avg=[COR_avg mean(cor2(:))];
    COR_avg_FOV=[COR_avg_FOV mean(cor2_FOV(:))];
    COR_avg_intensity=[COR_avg_intensity mean(cor2_intensity(:))];
    COR_avg_intensity_FOV=[COR_avg_intensity_FOV mean(cor2_intensity_FOV(:))];
    COR_avg_G=[COR_avg_G mean(cor2_G(:))];
    COR_avg_intensity_G=[COR_avg_intensity_G mean(cor2_intensity_G(:))];    
    MSE_avg=[MSE_avg mean(mean(mean((D-OUT2).^2)))/(255^2)];
    PSNR_avg=[PSNR_avg 10*log10(1/MSE_avg(end))];

COR_out=0;
MSE_out=0;
PSNR_out=0;

f7=fopen('train_dataF.bin','a+');
f8=fopen('train_labelsF.bin','a+');
f9=fopen('train_dataF_1.bin','a+');
f10=fopen('train_labelsF_1.bin','a+');
fwrite(f7,OUT2,'uint8');
fwrite(f8,B,'uint8');
fwrite(f9,D,'uint8');
fwrite(f10,B_sin,'uint8');
fclose(f7);
fclose(f8);
fclose(f9);
fclose(f10);


end



fid = fopen('AVG_corr_intensity.txt', 'a+');
fprintf(fid, '%d \n', mean(COR_avg_intensity));
fclose(fid);

fid = fopen('Avg_corr_intensity_G.txt', 'a+');
fprintf(fid, '%d \n', mean(COR_avg_intensity_G));
fclose(fid);

toc
