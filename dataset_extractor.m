function dataset_extractor
% This function extracts dataset from a collection of experimental MAT 
% files into 32bit TIFFs for all the captured absorption images you want to 
% use for the net training 

%% parameters

% Set the scaling parameters. You should set these parameters according to
% the distribution of your data. 
% The scaling parameters a and b are chosen such that to the range of the
% logarithm OD images will translate to a range within [0,1], including
% some safety boundaries to make sure that all the acquired data falls into 
% this range. Make sure that you use a wide enough range, otherwise some
% information will be lost during the saving process. However, the range 
% shouldn't be too wide to avoid lossy compression of information.
% You can use auto_find_a_b=1 to find them automatically.
% The a,b parameters rescale the range as [4.33, 9.67]*a+b=[0,1] in this
% example.

auto_find_a_b = 1;
a = 3/16;   % will be overridden if you use auto_find_a_b 
b = -13/16; % will be overridden if you use auto_find_a_b 

% The position of the center atomic cloud in the OD image, which will be
% used as the center of the image passed to DNN for analysis 
crop_TIFFs = 1; % if set to 0 the full frame (FOV) will be stored, this
                % will also work as the images will be alternatively 
                % cropped by the python script.
centerVer = 442.1312;
centerHor = 803.3961;

% Size of the input image to the net, centered around the atomic cloud
inL = 476;

% Range of cropping around the center
xRange = fix(centerVer +.5 +(-inL/2+1:inL/2));
yRange = fix(centerHor +.5 +(-inL/2+1:inL/2));


%% define relevant directories

% Home directory in which the extracted datasets folders will be saved
datasets_main_dir = 'C:/datasets/single_shot/'; 

% Create the relevant folders for saving the results
mkdir(datasets_main_dir)
mkdir([datasets_main_dir 'A_no_atoms'])
mkdir([datasets_main_dir 'R_no_atoms'])
mkdir([datasets_main_dir 'A_with_atoms'])
mkdir([datasets_main_dir 'R_with_atoms'])
mkdir([datasets_main_dir 'R_leftovers'])

% The main directory which contains data folders
data_main_dir = 'D:/Data 2019/';

% Subdirectories which contain data WITHOUT atoms
inDirs_no_atoms = {'2019_04_24/no_atoms/'
                   '2019_08_28/no_atoms/'
                   '2019_08_29/no_atoms/'
                   '2019_09_04/no_atoms/'
                   '2019_09_05/no_atoms/'
                   '2019_10_02/no_atoms/'
                   '2019_11_17/no_atoms/'
                   '2019_11_18/no_atoms/'
                   '2019_11_19/no_atoms/'
                   '2019_11_20/no_atoms/'
                   '2019_11_21/no_atoms/'};
               
% Subdirectories which contain data WITH atoms
inDirs_with_atoms = {'2019_04_24/atoms/'
                     '2019_08_28/atoms/'
                     '2019_08_29/atoms/'
                     '2019_09_04/atoms/'
                     '2019_09_05/atoms/'
                     '2019_09_05/atoms/'
                     '2019_10_02/atoms/'
                     '2019_11_17/atoms/'
                     '2019_11_18/atoms/'
                     '2019_11_19/atoms/'
                     '2019_11_20/atoms/'
                     '2019_11_21/atoms/'};


%% read images without atoms to compute a,b parameters
if auto_find_a_b
    cum_im_num = 0;

    for j = 1:length(inDirs_no_atoms)

    % list all the data files in the dataset & create a table for saving 
    % experimental metedata
        fList = dir([data_main_dir, inDirs_no_atoms{j}, 'exp*.mat']); 
        im_num = length(fList);
        cur_table = struct2table(fList);

        if j==1
            fT_no_atoms = cur_table;
        else
            cur_table.dark_images_path = cell(height(cur_table),1);
            cur_table.meanR = nan(height(cur_table),1);
            cur_table.stdR = nan(height(cur_table),1);
            cur_table.boundR_1 = nan(height(cur_table),1);
            cur_table.boundR_2 = nan(height(cur_table),1);
            cur_table.meanA = nan(height(cur_table),1);
            cur_table.stdA = nan(height(cur_table),1);
            cur_table.boundA_1 = nan(height(cur_table),1);
            cur_table.boundA_2 = nan(height(cur_table),1);
            cur_table.evap_cut = nan(height(cur_table),1);
            cur_table.time_of_flight = nan(height(cur_table),1);
            fT_no_atoms = [fT_no_atoms; cur_table]; %#ok<AGROW>
        end

    %   load the dark background images
        dark_images_path = split([data_main_dir, inDirs_no_atoms{j}], '/');
        dark_images_path = [char(join(dark_images_path(1:end-2), "/")) '/Program/dark_images.mat'];
        load(dark_images_path,'pixelfly_dark_2');

    %   iterate over all the images in the folder
        for id = 1:im_num
            cum_im_num = cum_im_num +1; % this is row idx in fTable

            load([data_main_dir inDirs_no_atoms{j} fList(id).name], 'pixelfly_image', 'last_evaporation_cut', 'delta_t_image');

            pixelfly_image = pixelfly_image - pixelfly_dark_2;
            A = pixelfly_image(1:1040,1:1392);
            R = pixelfly_image(1041:2080,1:1392);

    %       Calculate the logarithm of the OD image for the data and the reference image
            if crop_TIFFs
                R = real(log(R(xRange,yRange)));
                A = real(log(A(xRange,yRange)));
            else
                R = real(log(R));
                A = real(log(A));
            end

    %       save numerical parameters in the table
            fT_no_atoms.dark_images_path{cum_im_num} = dark_images_path;

    %       calculate image mean, std and calculate the bounds whithin which 1-1e-4 of the intensity is concentrated
            fT_no_atoms.meanR(cum_im_num) = mean(R(:));
            fT_no_atoms.stdR(cum_im_num) = std(R(:));
            cur_bounds = extract_bounds(R(:),1e-4);
            fT_no_atoms.boundR_1(cum_im_num) = cur_bounds(1);
            fT_no_atoms.boundR_2(cum_im_num) = cur_bounds(2);

            fT_no_atoms.meanA(cum_im_num) = mean(A(:));
            fT_no_atoms.stdA(cum_im_num) = std(A(:));
            cur_bounds = extract_bounds(A(:),1e-4);
            fT_no_atoms.boundA_1(cum_im_num) = cur_bounds(1);
            fT_no_atoms.boundA_2(cum_im_num) = cur_bounds(2);

    %       save physical parameters
            fT_no_atoms.evap_cut(cum_im_num) = last_evaporation_cut;
            fT_no_atoms.time_of_flight(cum_im_num) = delta_t_image;

            clearvars R A cur_bounds

    %       display process progress
            if ~mod(cum_im_num,10)
                disp([num2str(cum_im_num) ' no atoms images read in ' num2str(round(toc/60,1)) ' minutes'])
            end
        end

        disp(['dataset num = ',num2str(j) ' without atoms, proccessed in ' num2str(round(toc/60,1)) ' minutes'])
    end


end


%% display histograms of dataranges
figure
[y1, x1] = histcounts([fT_no_atoms.boundR_1; fT_no_atoms.boundA_1]);
lim1 = min([fT_no_atoms.boundR_1; fT_no_atoms.boundA_1])...
    - std([fT_no_atoms.boundR_1; fT_no_atoms.boundA_1]);
bar(mean([x1(1:end-1);x1(2:end)]), y1), hold on
[y2,x2] = histcounts([fT_no_atoms.boundR_2; fT_no_atoms.boundA_2]);
lim2 = max([fT_no_atoms.boundR_2; fT_no_atoms.boundA_2])...
    + std([fT_no_atoms.boundR_2; fT_no_atoms.boundA_2]);
bar(mean([x2(1:end-1);x2(2:end)]), y2)
ax = gca;
yL = ax.YLim;
plot(lim1*[1 1], yL)
plot(lim2*[1 1], yL)
ax.YLim = yL;
a = 1/(lim2-lim1);
b = -lim1*a;

legend('lower edge of images', 'upper edge of images', 'suggested lower bound',...
    'suggested upper bound')

disp(['suggested bounds are [' num2str(lim1) ',' num2str(lim2) '].'])
disp(['corresponding scaling parameterss are a=' num2str(a) '; b=' num2str(b) ';'])


%% extract TIFF files for images without atoms
cum_im_num = 0;

for j = 1:length(inDirs_no_atoms)
    
% list all the data files in the dataset & create a table for saving 
% experimental metedata
    fList = dir([data_main_dir, inDirs_no_atoms{j}, 'exp*.mat']); 
    im_num = length(fList);
    cur_table = struct2table(fList);
    
    if j==1
        fT_no_atoms = cur_table;
    else
        cur_table.dark_images_path = cell(height(cur_table),1);
        cur_table.meanR = nan(height(cur_table),1);
        cur_table.stdR = nan(height(cur_table),1);
        cur_table.boundR_1 = nan(height(cur_table),1);
        cur_table.boundR_2 = nan(height(cur_table),1);
        cur_table.meanA = nan(height(cur_table),1);
        cur_table.stdA = nan(height(cur_table),1);
        cur_table.boundA_1 = nan(height(cur_table),1);
        cur_table.boundA_2 = nan(height(cur_table),1);
        cur_table.evap_cut = nan(height(cur_table),1);
        cur_table.time_of_flight = nan(height(cur_table),1);
        fT_no_atoms = [fT_no_atoms; cur_table]; %#ok<AGROW>
    end
    
%   load the dark background images
    dark_images_path = split([data_main_dir, inDirs_no_atoms{j}], '/');
    dark_images_path = [char(join(dark_images_path(1:end-2), "/")) '/Program/dark_images.mat'];
    load(dark_images_path,'pixelfly_dark_2');
    
%   iterate over all the images in the folder
    for id = 1:im_num
        cum_im_num = cum_im_num +1; % this is row idx in fTable
        
        load([data_main_dir inDirs_no_atoms{j} fList(id).name], 'pixelfly_image', 'last_evaporation_cut', 'delta_t_image');
        
        pixelfly_image = pixelfly_image - pixelfly_dark_2;
        A = pixelfly_image(1:1040,1:1392);
        R = pixelfly_image(1041:2080,1:1392);
               
%       Calculate the logarithm of the OD image for the data and the reference image
        if crop_TIFFs
            R = real(log(R(xRange,yRange)));
            A = real(log(A(xRange,yRange)));
        else
            R = real(log(R));
            A = real(log(A));
        end
        
%       rescale the images to the appropriate bounds and save TIFF files
        A = a*A + b;
        R = a*R + b;
        tiffwrite32bit(uint32(4294967295*A), [datasets_main_dir 'A_no_atoms\' num2str(cum_im_num,'%06.f') '.tif'])
        tiffwrite32bit(uint32(4294967295*R), [datasets_main_dir 'R_no_atoms\' num2str(cum_im_num,'%06.f') '.tif'])
        
%       save numerical parameters in the table
        fT_no_atoms.dark_images_path{cum_im_num} = dark_images_path;
        
%       calculate image mean, std and calculate the bounds whithin which 1-1e-4 of the intensity is concentrated
        fT_no_atoms.meanR(cum_im_num) = mean(R(:));
        fT_no_atoms.stdR(cum_im_num) = std(R(:));
        cur_bounds = extract_bounds(R(:),1e-4);
        fT_no_atoms.boundR_1(cum_im_num) = cur_bounds(1);
        fT_no_atoms.boundR_2(cum_im_num) = cur_bounds(2);
        
        fT_no_atoms.meanA(cum_im_num) = mean(A(:));
        fT_no_atoms.stdA(cum_im_num) = std(A(:));
        cur_bounds = extract_bounds(A(:),1e-4);
        fT_no_atoms.boundA_1(cum_im_num) = cur_bounds(1);
        fT_no_atoms.boundA_2(cum_im_num) = cur_bounds(2);
        
%       save physical parameters
        fT_no_atoms.evap_cut(cum_im_num) = last_evaporation_cut;
        fT_no_atoms.time_of_flight(cum_im_num) = delta_t_image;
        
        clearvars R A cur_bounds
        
%       display process progress
        if ~mod(cum_im_num,10)
            disp([num2str(cum_im_num) ' no atoms TIFFs saved in ' num2str(round(toc/60,1)) ' minutes'])
        end
    end
    
    disp(['dataset num = ',num2str(j) ' without atoms, proccessed in ' num2str(round(toc/60,1)) ' minutes'])
end

% save the table containing data analysis information 
save([datasets_main_dir 'fTables'], 'fT_no_atoms')


%% extract TIFF files for with atoms

cum_im_with = 0;
cum_im_left = 0;
fT_with_atoms = table;
fT_leftovers = table;
tic

for j = 1:length(inDirs_with_atoms)
        
%   list all the data files in the dataset & create a table for saving physical observables and image properties
    fList = dir([data_main_dir, inDirs_with_atoms{j}, 'temp*.*']); 
    im_num = length(fList);
    cur_table = struct2table(fList);
    
    dark_images_path = split([data_main_dir, inDirs_with_atoms{j}], '/');
    dark_images_path = [char(join(dark_images_path(1:end-2), "/")) '/Program/dark_images.mat'];
    load(dark_images_path,'pixelfly_dark_2');
    
    for id = 1:im_num
        load([data_main_dir inDirs_with_atoms{j} fList(id).name], 'pixelfly_image', 'last_evaporation_cut', 'delta_t_image',...
            'center_x', 'center_y');
        
%       check the format of the data image   

        pixelfly_image = pixelfly_image - pixelfly_dark_2;
        A = pixelfly_image(1:1040,1:1392);
        R = pixelfly_image(1041:2080,1:1392);
        
%       Calculate the logarithm of the OD image for the data and the reference image
        if crop_TIFFs
            R = real(log(R(xRange,yRange)));
            A = real(log(A(xRange,yRange)));
        else
            R = real(log(R));
            A = real(log(A));
        end
        
%       Find the area occupied by the atomic cloud
        dist = sqrt((center_x -centerVer).^2 +(center_y -centerHor).^2);
        atomsFlag = (dist<10);
        
        if atomsFlag %data contains atoms, therefore store the atomic and the reference frame
            
            cum_im_with = cum_im_with +1; %count the number of data with atoms
            
%           rescale the images to the appropriate bounds and save TIFF files
            A = a*A + b;
            R = a*R + b;
            tiffwrite32bit(uint32(4294967295*A), [datasets_main_dir 'A_with_atoms/' num2str(cum_im_with,'%06.f') '.tif'])
            tiffwrite32bit(uint32(4294967295*R), [datasets_main_dir 'R_with_atoms/' num2str(cum_im_with,'%06.f') '.tif'])
            
%           save numerical parameters in the table
            fT_with_atoms.name{cum_im_with} = cur_table.name{id};
            fT_with_atoms.folder{cum_im_with} = cur_table.folder{id};
            fT_with_atoms.date{cum_im_with} = cur_table.date{id};
            fT_with_atoms.bytes(cum_im_with) = cur_table.bytes(id);
            
            fT_with_atoms.dark_images_path{cum_im_with} = dark_images_path;
            fT_with_atoms.meanR(cum_im_with) = mean(R(:));
            fT_with_atoms.stdR(cum_im_with) = std(R(:));
            cur_bounds = extract_bounds(R(:),1e-4);
            fT_with_atoms.boundR_1(cum_im_with) = cur_bounds(1);
            fT_with_atoms.boundR_2(cum_im_with) = cur_bounds(2);
            
            fT_with_atoms.meanA(cum_im_with) = mean(A(:));
            fT_with_atoms.stdA(cum_im_with) = std(A(:));
            cur_bounds = extract_bounds(A(:),1e-4);
            fT_with_atoms.boundA_1(cum_im_with) = cur_bounds(1);
            fT_with_atoms.boundA_2(cum_im_with) = cur_bounds(2);
            
            fT_with_atoms.evap_cut(cum_im_with) = last_evaporation_cut;
            fT_with_atoms.time_of_flight(cum_im_with) = delta_t_image;
            fT_with_atoms.cx(cum_im_with) = center_x;
            fT_with_atoms.cy(cum_im_with) = center_y;
            
        else % no atoms to use in this data, store only the reference image
            
            cum_im_left = cum_im_left +1;
            
            R = a*R + b;
            tiffwrite32bit(uint32(4294967295*R), [datasets_main_dir 'R_leftovers/' num2str(cum_im_left,'%06.f') '.tif'])
            
            fT_leftovers.name{cum_im_left} = cur_table.name{id};
            fT_leftovers.folder{cum_im_left} = cur_table.folder{id};
            fT_leftovers.date{cum_im_left} = cur_table.date{id};
            fT_leftovers.bytes(cum_im_left) = cur_table.bytes(id);
            
            fT_leftovers.dark_images_path{cum_im_left} = dark_images_path;
            fT_leftovers.meanR(cum_im_left) = mean(R(:));
            fT_leftovers.stdR(cum_im_left) = std(R(:));
            cur_bounds = extract_bounds(R(:),1e-4);
            fT_leftovers.boundR_1(cum_im_left) = cur_bounds(1);
            fT_leftovers.boundR_2(cum_im_left) = cur_bounds(2);
            
            fT_leftovers.meanA(cum_im_left) = mean(A(:));
            fT_leftovers.stdA(cum_im_left) = std(A(:));
            cur_bounds = extract_bounds(A(:),1e-4);
            fT_leftovers.boundA_1(cum_im_left) = cur_bounds(1);
            fT_leftovers.boundA_2(cum_im_left) = cur_bounds(2);
            
            fT_leftovers.evap_cut(cum_im_left) = last_evaporation_cut;
            fT_leftovers.time_of_flight(cum_im_left) = delta_t_image;
        end
        
        clearvars R A cur_bounds

%       display progress
        if ~mod(cum_im_with+cum_im_left,10)
            disp([num2str(cum_im_with+cum_im_left) ' with atoms tiffs saved in ' num2str(round(toc/60,1)) ' minutes'])
        end
    end
    disp(['dataset num = ',num2str(j) ' with atoms, proccessed in ' num2str(round(toc/60,1)) ' minutes']) 
end

% save the table containing data analysis information 
save([datasets_main_dir 'fTables'], 'fT_with_atoms', 'fT_leftovers', '-append')


end

function tiffwrite32bit(imdata, fname)
    t = Tiff(fname,'w');
    t.setTag('ImageLength',size(imdata,1));
    t.setTag('ImageWidth', size(imdata,2));
    t.setTag('Photometric', Tiff.Photometric.MinIsBlack);
    t.setTag('BitsPerSample', 32);
    t.setTag('SamplesPerPixel', 1);
    t.setTag('Compression', Tiff.Compression.None);
    t.setTag('PlanarConfiguration', Tiff.PlanarConfiguration.Chunky);
    t.setTag('Software', 'MATLAB');
    t.write(imdata);
    t.close();
end

function bounds = extract_bounds(in, frac)

    [~,ord] = sort(in(:));
    n = length(in(:));
    crop = min(1+subplus(round(n*frac-1)),n);
    bounds = (ord(crop))*[1 1];
    bounds(2) = ord(end-crop);
    bounds = in(bounds);
end
