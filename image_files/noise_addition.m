% noise addition
Files=dir('*.*');  % change this to directory where encrypted image files exist
noise = rand([256, 256]);
w = [0.25, 0.5, 1];  % value of noise addition
for j=1:1:3
    p = w(j);
    for k=3:length(Files)-1
        FileNames = Files(k).name;
        E = load(FileNames);
        E = E.E;

        Red = E(:,:,1);
        Green = E(:,:,2);
        Blue = E(:,:,3); 

        Red = Red.*(1+(p*noise));
        Green = Green.*(1+(p*noise));
        Blue = Blue.*(1+(p*noise));
        E = cat(3, Red, Green, Blue);

        A = abs(E);
        P = angle(E);
        if p==0.25
            amp_path = 'image_files/Noise_0.25/Amp/';
            phase_path = 'image_files/Noise_0.25/Phase/';
        elseif p==0.5
            amp_path = 'image_files/Noise_0.5/Amp/';
            phase_path = 'image_files/Noise_0.5/Phase/';
        else
            amp_path = 'image_files/Noise_1/Amp/';
            phase_path = 'image_files/Noise_1/Phase/';
        end

        str_a = [amp_path,FileNames];
        str_p = [phase_path, FileNames];
       
        save(str_a,'A')
        save(str_p,'P')
    end
end
