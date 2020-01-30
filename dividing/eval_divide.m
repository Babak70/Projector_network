
NUMEVAL_SAMPLES=20000;
HEIGHT2=200;



HEIGHT=51;
NUMEVAL_SAMPLES2=1000;
WIDTH2=HEIGHT2;
WIDTH=HEIGHT;
n1=floor(NUMEVAL_SAMPLES/NUMEVAL_SAMPLES2);
res=NUMEVAL_SAMPLES-n1*NUMEVAL_SAMPLES2;


f2=fopen('./train_dataF_1.bin','r');
A2=fread(f2,HEIGHT2*WIDTH2*NUMEVAL_SAMPLES,'uint8');
C=reshape(A2,[HEIGHT2,WIDTH2,NUMEVAL_SAMPLES]);
fclose(f2);


for i=1:n1
    str=['./dividing\eval_labelsF_' num2str(i) '.bin']
    fullfile(str)
    str2=['./dividing\eval_dataF_' num2str(i) '.bin']
    V=(1+(i-1)*NUMEVAL_SAMPLES2):((i)*NUMEVAL_SAMPLES2);
    
    f=fopen(str,'w'); 
    fwrite(f,zeros(WIDTH,HEIGHT,length(V))*255,'uint8');
    fclose(f);
    
    g=fopen(str2,'w'); 
    fwrite(g,C(:,:,V),'uint8');
    fclose(g);
end


%%res:
V=(1+(n1)*NUMEVAL_SAMPLES2):NUMEVAL_SAMPLES;



for i=1:n1
    str=['./dividing\eval_labelsFf_' num2str(i) '.bin']
    str2=['./dividing\eval_dataFf_' num2str(i) '.bin']
    V=(1+(i-1)*NUMEVAL_SAMPLES2):((i)*NUMEVAL_SAMPLES2);
    
    f=fopen(str,'w'); 
    fwrite(f,zeros(WIDTH,HEIGHT,length(V))*255,'uint8');
    fclose(f);
    
    g=fopen(str2,'w'); 
    fwrite(g,C(:,:,V),'uint8');
    fclose(g);
end
V=(1+(n1)*NUMEVAL_SAMPLES2):NUMEVAL_SAMPLES;


