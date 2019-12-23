clear all
close all
clc
L = 4;
% K =[-1, 5, 5, -1]/8;
% K =[1, 1, 1, 1]/4;
% K = [1 -2 1 0];
K = [-1 3 -1 0];        %Sharpening Mask
% K = [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025];    %Decomp. lpf
% K =[-0.48296291314469025, 0.836516303737469, -0.22414386804185735, -0.12940952255092145]; %Decomp. hpf
% K =[0.48296291314469025, 0.836516303737469, 0.22414386804185735, -0.12940952255092145];   %Recon lpf
% K = [-0.12940952255092145, -0.22414386804185735, 0.836516303737469, -0.48296291314469025]; %Recon hpf

n = 5;

src_dir = 'E:\RnD\Current_Projects\Musawwir\Frameworks\SW\Dataset\Test_Vid_Sequences\yuv\';
file_names = {'foreman_cif.yuv','akiyo_cif.yuv','bridge-close_cif.yuv','bus_cif.yuv','carphone_cif.yuv','coastguard_cif.yuv','hall_cif.yuv','mobile_cif.yuv','news_cif.yuv','football_cif.yuv','stefan_cif.yuv','tempete_cif.yuv','waterfall_cif.yuv','container_cif.yuv'};
snr = zeros(14,1);
e_0 = 0;
e_4 = 0;
e_8 = 0;
e_16 = 0;

for ii=6:6,
    close all
    [y, u, v] = yuvRead([src_dir,file_names{ii}], 352, 288, 1);
    img = y;
    % img = rgb2gray(imread('E:\RnD\Current_Projects\Musawwir\Frameworks\SW\Dataset\Test_Images\circle00.bmp'));
    % img = ones(352,288)*255;
    % img(100:200,10:200) = 0;
    imshow(uint8(img))
    title('Original Image')

    % out1 = FastFIR_DF(img,L);
    % figure
    % imshow(out1)
%     out2 = FastFIR_TF(img,L);
%     figure
%     imshow(out2)
    out3 = FastFIR_TF_Mod(img,L,K);
%     out3 = DECOR(img,L,K,n);
    figure
    imshow(out3)
    title('Filtered Image - Proposed')
    
    filt_img = uint8(filter2(K,img));
    %compensate for shifting due to convolution, can be different for different kernels, [0,0,-1,-1,1,-1,-1,1]
    filt_img = circshift(filt_img,floor(length(K)*0.5)+1,2);
    figure
    imshow(filt_img)
    title('Filtered Image - Conventional')

    % e1 = double(filt_img(:,4:end))-double(out1(:,4:end));
    % e2 = double(filt_img(:,4:end))-double(out2(:,4:end));
    e3 = double(filt_img(:,4:end))-double(out3(:,4:end));
    % sum(sum(abs(e1)))
    % sum(sum(abs(e2)))
    sum(sum(abs(e3)));

    sp = sum(sum(double(filt_img).^2));
    % np1 = sum(sum(e1.^2));
    % np2 = sum(sum(e2.^2));
    np3 = sum(sum(e3.^2));
    % snr1 = 10*log10(sp/np1)
    % snr2 = 10*log10(sp/np2)
    snr3 = 10*log10(sp/np3)
    snr(ii) = snr3;
    
    e3 = e3(:);
    figure
    [h,b] = hist(e3,100);
    h = h/(352*288);
    error_free_pixels = max(max(h));
    e_0 = e_0 + error_free_pixels;%sum(abs(e3)<=0)/(352*288)
    e_4 = e_4 + sum(abs(e3)<=4)/(352*288)
    e_8 = e_8 + sum(abs(e3)<=8)/(352*288)
    e_16 = e_16 + sum(abs(e3)<=16)/(352*288)
    c=bar(b,h);
    c.FaceColor(:,:) = [0.5 0.5 0.5];
    c.EdgeColor(:,:) = [0.5 0.5 0.5];
    
    xlim([-128,127])
    set(gca,'yscale','log')
    set(gca,'FontSize',35)
    xlabel('Error Value')
    ylabel('Probability')
    txt = {['\leftarrow',num2str(error_free_pixels*100),'% pixels with zero error' newline '    SNR = ' num2str(snr3)]};
    text(0,0.125,txt,'FontSize',30)
    title('Foreman - K1')
%     e3 = double(e3==0);
%     100*sum(e3)/length(e3)
%     pause
end

e_0 = e_0/14
e_4 = e_4/14
e_8 = e_8/14
e_16 = e_16/14

%Modified Transposed form Fast FIR parallel filter
function out = FastFIR_TF_Mod(img,L,K)
img = double(img);
[row,col] = size(img);
out = zeros(size(img));
x = zeros(1,L);
xz_L = zeros(1,L);

h0 = K(1);
h1 = K(2);
h2 = K(3);
h3 = K(4);

for r=1:row,
    for c=1:L:col,
        x = img(r,c:c+L-1);
        x = fliplr([x(1),x(3),x(2),x(4)]);
        
        Q1 = x(1)+x(2)-x(3)-x(4);
        Q2 = x(2)-x(4);
        Q3 = xz_L(1)+x(2)-xz_L(3)-x(4);
        Q4 = x(3)-x(4);
        Q5 = x(4);
        Q6 = xz_L(3)-x(4);
        Q7 = xz_L(1)+x(2)-x(3)-x(4);
        Q8 = xz_L(1)-x(4);
        Q9 = xz_L(1)+xz_L(2)-xz_L(3)-x(4);
        
        %Saturate low entropy signals
        if(Q1<-32) Q1 = -32; end
        if(Q1>31) Q1 = 31; end
        if(Q3<-32) Q3 = -32; end
        if(Q3>31) Q3 = 31; end
        if(Q7<-32) Q7 = -32; end
        if(Q7>31) Q7 = 31; end
        if(Q9<-32) Q9 = -32; end
        if(Q9>31) Q9 = 31; end
    
        F1 = h0*Q1;
        F2 = (h2-h0)*Q2;
        F3 = h2*Q3;
        F4 = (h0+h1)*Q4;
        F5 = (h0+h1+h2+h3)*Q5;
        F6 = (h2+h3)*Q6;
        F7 = h1*Q7;
        F8 = (h3-h1)*Q8;
        F9 = h3*Q9;
                
        y3 = F1+F2+F4+F5;
        y1 = -F2+F3+F5+F6;
        y2 = F4+F5+F7+F8;
        y0 = F5+F6-F8+F9;

        out(r,c:c+L-1) = [y0,y1,y2,y3];

        xz_L = x;
    end
end
out = uint8(out);
end

%Transposed form Fast FIR parallel filter
function out = FastFIR_TF(img,L)
img = double(img);
[row,col] = size(img);
out = zeros(size(img));
x = zeros(1,L);
xz_L = zeros(1,L);

h0 = 0.25;
h1 = 0.25;
h2 = 0.25;
h3 = 0.25;

for r=1:row,
    for c=1:L:col,
        x = img(r,c:c+L-1);
        x = fliplr([x(1),x(3),x(2),x(4)]);
        
        Q1 = x(1)-x(2)-x(3)+x(4);
        Q2 = x(2)-x(4);
        Q3 = xz_L(1)-x(2)-xz_L(3)+x(4);
        Q4 = x(3)-x(4);
        Q5 = x(4);
        Q6 = xz_L(3)-x(4);
        Q7 = -xz_L(1)+x(2)-x(3)+x(4);
        Q8 = xz_L(1)-x(4);
        Q9 = -xz_L(1)+xz_L(2)-xz_L(3)+x(4);
    
        F1 = h0*Q1;
        F2 = (h2+h0)*Q2;
        F3 = h2*Q3;
        F4 = (h0+h1)*Q4;
        F5 = (h0+h1+h2+h3)*Q5;
        F6 = (h2+h3)*Q6;
        F7 = h1*Q7;
        F8 = (h3+h1)*Q8;
        F9 = h3*Q9;
                
        y3 = F1+F2+F4+F5;
        y1 = F2+F3+F5+F6;
        y2 = F4+F5+F7+F8;
        y0 = F5+F6+F8+F9;

        out(r,c:c+L-1) = [y0,y1,y2,y3];

        xz_L = x;
    end
end
out = uint8(out);
end

%Direct form Fast FIR parallel filter
function out = FastFIR_DF(img,L)
img = double(img);
[row,col] = size(img);
out = zeros(size(img));
x = zeros(1,L);
F = zeros(1,9);
Fz_L = zeros(1,9);
h0 = 0.25;
h1 = 0.25;
h2 = 0.25;
h3 = 0.25;

for r=1:row,
    for c=1:L:col,
        x = img(r,c:c+L-1);
        x = [x(1),x(3),x(2),x(4)];
        
        P1 = x(1);
        P2 = x(1)+x(2);
        P3 = x(2);
        P4 = x(1)+x(3);
        P5 = x(1)+x(2)+x(3)+x(4);
        P6 = x(2)+x(4);
        P7 = x(3);
        P8 = x(3)+x(4);
        P9 = x(4);
    
        F1 = h0*P1;
        F2 = (h2+h0)*P2;
        F3 = h2*P3;
        F4 = (h0+h1)*P4;
        F5 = (h0+h1+h2+h3)*P5;
        F6 = (h2+h3)*P6;
        F7 = h1*P7;
        F8 = (h3+h1)*P8;
        F9 = h3*P9;
        F = [F1,F2,F3,F4,F5,F6,F7,F8,F9];
        
        y0 = F(1)+Fz_L(3)-Fz_L(7)+Fz_L(8)-Fz_L(9);
        y2 = -F(1)+F(2)-F(3)+F(7)+Fz_L(9);
        y1 = -F(1)-Fz_L(3)+F(4)+Fz_L(6)-F(7)-Fz_L(9);
        y3 = F(1)-F(2)+F(3)-F(4)+F(5)-F(6)+F(7)-F(8)+F(9);

        out(r,c:c+L-1) = [y0,y1,y2,y3];

        Fz_L = F;
    end
end
out = uint8(out);
end

%DECOR filter
function out = DECOR(img,L,K,n)
img = double(img);
[row,col] = size(img);
out = zeros(size(img));
xz = zeros(1,length(K));    %Original Values
xzd = zeros(1,length(K));   %Differences
yz = zeros(1,L);

for r=1:row,
    for c=1:col,
%         xz = circshift(xz,1,2);
%         xz(1) = img(r,c);
%         out(r,c) = sum(sum(K.*xz)); 

        xz = circshift(xz,1,2);
        xz(1) = img(r,c);
        xzd = circshift(xzd,1,2);
%         xzd(1) = img(r,c)-xz(2)+xz(3)-xz(4);
        xzd(1) = img(r,c)-xz(2);
        %Saturate low entropy signals
        if(xzd(1)<-2^n) xzd(1) = -2^n; end
        if(xzd(1)>(2^n)-1) xzd(1) = (2^n)-1; end
       
        yz = circshift(yz,1,2);
%         out(r,c) = sum(sum(K.*xzd))+yz(2)-yz(3)+yz(4);
        out(r,c) = sum(sum(K.*xzd))+yz(2);
        yz(1) = out(r,c);
        if(yz(1)>255) yz(1) = 255; end
        if(yz(1)<0) yz(1) = 0; end
    end
end
out = uint8(out);
end