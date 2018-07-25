% img_rootpath = 'C:/Datasets/CASME2_Cropped/';
% out_rootpath = 'E:/Documents/TIM/TIM/output/';

pkg load image

img_rootpath = '/home/mihag/Documents/CASME2_Cropped/';
out_rootpath = '/home/mihag/Documents/CASME2_TIM10_Color/';

subject_dir = dir(img_rootpath);
subject_dir = subject_dir(3:length(subject_dir));

for i=1: length(subject_dir)
    mkdir([out_rootpath, subject_dir(i).name]);
    vid_dir = dir( [ img_rootpath, subject_dir(i).name ] );
    
    vid_dir = vid_dir(3:length(vid_dir));
    
    for j=1: length(vid_dir)
        mkdir([out_rootpath, subject_dir(i).name, '/' ,  vid_dir(j).name])
    end
end

for i=1: length(subject_dir)
    vid_dir = dir( [ img_rootpath, subject_dir(i).name ] );
    vid_dir = vid_dir(3:length(vid_dir));
    
    for j=1: length(vid_dir)
       image_path = [img_rootpath, subject_dir(i).name, '/' ,  vid_dir(j).name];
       out_path = [out_rootpath, subject_dir(i).name, '/' ,  vid_dir(j).name];
       tim_animate(image_path, out_path, 10);
    end
    
    disp([int2str(i), '/', int2str(length(subject_dir)), ' done!']);
end

%% denoising

% img_rootpath = 'C:/Users/ViperCPU/Documents/CASME2_Output_a10/';
% out_rootpath = 'C:/Users/ViperCPU/Documents/CASME2_Output_a10/';
% 
% subject_dir = dir(img_rootpath);
% subject_dir = subject_dir(3:length(subject_dir));
% 
% for i=1: length(subject_dir)
%     vid_dir = dir( [ img_rootpath, subject_dir(i).name ] );
%     vid_dir = vid_dir(3:length(vid_dir));
%     
%     for j=1: length(vid_dir)
%        image_path = [img_rootpath, subject_dir(i).name, '/' ,  vid_dir(j).name];
% %        out_path = [out_rootpath, subject_dir(i).name, '/' ,  vid_dir(j).name];
%         file_dir = dir(image_path);
%         file_dir = file_dir(3:length(file_dir));
%         
%         for k=1 : length(file_dir)
%             input = [image_path, '/', file_dir(k).name];
%             
% %             if k < 10
% %                 num_string = ['00', int2str(k), '.jpg'];
% %             else
% %                 num_string = ['0', int2str(k), '.jpg'];
% %             end
%             out = [out_rootpath, subject_dir(i).name, '/', vid_dir(j).name, '/', file_dir(k).name];
%             img = imread(input);
%             img = wiener2(img, [5, 5]);
% %             imshow(img);
%             imwrite(img, out);
%         end
%         
%     end
%     
%     disp([int2str(i), '/', int2str(length(subject_dir)), ' done!']);
%     
% end
% 
% 
% 
% 
% 
% 
