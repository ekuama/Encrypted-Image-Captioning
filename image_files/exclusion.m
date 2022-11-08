% exclusion
Files=dir('*.*');  % change this to directory where encrypted image files exist
w = [13108, 26215, 39322];  % number of pixels equal to zero
for j=1:1:3
    p = randperm(256*256,w(j));
    for k=3:length(Files)-2
        FileNames = Files(k).name;
        E = load(FileNames);
        E = E.E;

        Red = E(:,:,1);
        Green = E(:,:,2);
        Blue = E(:,:,3); 

        Red(p) = 0;
        Green(p) = 0;
        Blue(p) = 0;
        E = cat(3, Red, Green, Blue);

        A = abs(E);
        P = angle(E);
        if j==1
            amp_path = 'image_files/Exclude_20/Amp/';
            phase_path = 'image_files/Exclude_20/Phase/';
        elseif j==2
            amp_path = 'image_files/Exclude_40/Amp/';
            phase_path = 'image_files/Exclude_40/Phase/';
        else
            amp_path = 'image_files/Exclude_60/Amp/';
            phase_path = 'image_files/Exclude_60/Phase/';
        end

        str_a = [amp_path,FileNames];
        str_p = [phase_path, FileNames];
       
        save(str_a,'A')
        save(str_p,'P')
    end
end
