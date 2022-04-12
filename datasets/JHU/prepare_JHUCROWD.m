%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code to generate density maps for JHU-CROWD++ dataset.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all;
maxSize = [2048,2048];
minSize = [512,512];
mode = 'train';
path ='/workspace/data/jhu_crowd_v2.0/';
output_path = strcat('/workspace/home/jourdanfa/data/density_maps/jhu_crowd_v2.0/', mode);
train_path_img = strcat(output_path,'/', 'images/');
train_path_den = strcat(output_path,'/', 'den/');
train_path_gt= strcat(output_path,'/', 'gt/');

mkdir(output_path);
mkdir(train_path_img);
mkdir(train_path_den);
mkdir(train_path_gt);

gt_list = dir(strcat(path, mode, '/gt/'));

if (strcmp(mode,'train') == 1 )
    num_files = 2272;
elseif (strcmp(mode,'val') == 1 )
    num_files = 500;
elseif (strcmp(mode,'test') == 1 )
    num_files = 1600;
end

avg = [];
stdev = [];
for idx = 1:num_files
    i = idx;
    if (mod(idx,10)==0)
        fprintf(1,'Set: Processing %3d/%d files\n', idx, num_files);
    end
    filename = replace(gt_list(idx+2).name,'.txt','');
 
    image_info = load(strcat(path, mode, '/gt/', filename,'.txt')) ;
    input_img_name = strcat(path,mode,  '/images/',filename,'.jpg');
    im = imread(input_img_name);  
    if (length(size(im)) == 2)
       [h, w, c] = size(im);
       im_tmp = im;
       im = zeros(h,w,3);
       im(:,:,1) = im_tmp;
       im(:,:,2) = im_tmp;
       im(:,:,3) = im_tmp;
       im = uint8(im);
    end
    [h, w, c] = size(im);
    
    im_reshaped = reshape(im,[h*w,3]);    
    avg = [avg; mean(im_reshaped)];
    stdev = [stdev; std(double(im_reshaped))];
 

    if (h > maxSize(1)) || ( w > maxSize(2))
        rate = maxSize(1)/h;
        rate_w = w*rate;
        if rate_w>maxSize(2)
            rate = maxSize(2)/w;
        end        
        im = imresize(im,[int16(h*rate),int16(w*rate)]);
    elseif (h < minSize(1)) || ( w < minSize(2))
        rate = minSize(1)/h;
        rate_w = w*rate;
        if rate_w<minSize(2)
            rate = minSize(2)/w;
        end        
        im = imresize(im,[int16(h*rate),int16(w*rate)]);
        
    else
        rate = 1; 
    end
    
    if (isempty(image_info))
        annPoints  = [];
    else
        annPoints =  image_info(:,1:2);
        annPoints(:,1) = annPoints(:,1)*double(rate);
        annPoints(:,2) = annPoints(:,2)*double(rate);    
 
    end
    
    im_density = get_density_map_gaussian(im,annPoints,15,4); 
    im_density = im_density(:,:,1);
    
    imwrite(im, [train_path_img filename '.jpg']);
    csvwrite([train_path_den filename '.csv'], im_density);
    csvwrite([train_path_gt filename '.txt'], annPoints);
end


 