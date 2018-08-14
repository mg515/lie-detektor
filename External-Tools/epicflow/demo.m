



path = '/home/miha/Documents/ME_data/CASME2_Color_TIM10/';
path_org = [path, 'CASME2_Color_TIM10']
sub_folders = rdir(path_org);
flow_folder = [path, '/flow/'];
png_folder = [path, '/png/'];
addpath(genpath('utils'));

sub_folders1 = {sub_folders.name}

for sub = sub_folders1
    videos = rdir(char(sub));
    videos = {videos.name};
    for video = videos
        video = char(video);
        split = strsplit(video, '/')
        save_dir = strjoin([flow_folder, string(split(end-1)), '/', string(split(end)), '/'], '');
        dir_get_epicflow(char(video), char(save_dir));
        
        flows = rdir(char(save_dir))
        flows = {flows.name}
        for flow = flows
            flow = char(flow)
            flow_split = strsplit(flow, '/')

            save_flow_dir = char([png_folder, char(flow_split(end-2)), '/', char(flow_split(end-1)), '/'])
            mkdir(save_flow_dir)
            save_flow_file = char([save_flow_dir,  char(flow_split(end)), '.png'])

            imwrite(flowToColor(readFlowFile(flow)), save_flow_file)
        end
        
        
    end

    
end

