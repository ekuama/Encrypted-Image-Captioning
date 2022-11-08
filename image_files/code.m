% clc, clear
Files=dir('*.*');  % change this to directory where original image files exist

n_x=rand([256, 256]); % random uniform distribution between the range of 0 and 1
b_x=rand([256, 256]); % random uniform distribution between the range of 0 and 1

for k=3:length(Files)-2
    FileNames = Files(k).name;
    display(FileNames)
    I=imread(FileNames);
    I=im2double(I);
    I = imresize(I, [256,256], 'bilinear');
    figure; imshow(I); axis off
    [Red, Green, Blue] = imsplit(I);
    
    Red = ifft2(fft2(Red.*exp(2*pi*1i*n_x)).*exp(2*pi*1i*b_x));
    Green = ifft2(fft2(Green.*exp(2*pi*1i*n_x)).*exp(2*pi*1i*b_x));
    Blue = ifft2(fft2(Blue.*exp(2*pi*1i*n_x)).*exp(2*pi*1i*b_x));
    E = cat(3, Red, Green, Blue);
    A = abs(E);
    P = angle(E);
	
    resize_path = 'image_files/Resize/';
    encrypt_path = 'image_files/Encrypt/';
    amp_path = 'image_files/Amp/';
    phase_path = 'image_files/Phase/';
	
    str_a = [amp_path,FileNames];
    str_p = [phase_path, FileNames];
    str_e = [encrypt_path, FileNames];
    str_r = [resize_path, FileNames];
	
    save([str_r  '.mat'],'I')
    save([str_e  '.mat'],'E')
    save([str_a '.mat'],'A')
    save([str_p '.mat'],'P')
end

save('image_files/rpm1.mat', 'n_x')
save('image_files/rpm2.mat', 'b_x')