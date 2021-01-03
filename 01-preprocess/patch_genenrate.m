data_path = 'G:\DataBase\02_ZS_HCC_pathological\02-data-block\01-bbox-patch';
dst_path = 'G:\DataBase\02_ZS_HCC_pathological\02-data-block\02-img-tiles';

sizeX = 512;
sizeY = 512;
slide_step = 512;

all_patient_files = dir(data_path);

for i = 3:length(all_patient_files)
    start = tic;
    patient_id = all_patient_files(i).name;
    patient_dir = [data_path, '\', patient_id];
    all_types = dir([patient_dir, '\', '#*']);
    for j = 1:length(all_types)
        type = all_types(j).name;  % # aa5500
        data_type_path = [patient_dir, '\', type];
        all_mat_files = dir(data_type_path);
        dst_type_path = [dst_path, '\', type];
        if ~exist(dst_type_path)
            mkdir(dst_type_path)
        end
        for m = 3:length(all_mat_files)
            mat_path = [data_type_path, '\', all_mat_files(m).name];
            mat = load(mat_path);
            mask = mat.mask;
            img = mat.img;
            % normalize mask
            switch type
                case '#000000'
                    mask = mask / 1;
                case '#005500'
                    mask = mask / 2;
                case '##aa5500'
                    mask = mask / 3;
                case '#00aa00'
                    mask = mask / 4;
                case '#aaaa00'
                    mask = mask / 5;
                case '#ff0000'
                    mask = mask / 6;
                case '#ffff00'
                    mask = mask / 7;
                case '#ffaa7f'
                    mask = mask / 8;
                case '#00ffff'
                    mask = mask / 9;
                case '#ff55ff'
                    mask = mask / 10;
            end
            patch_base_name = [dst_type_path, '\', patient_id, '_', all_mat_files(m).name];
            slide_box(img, mask, sizeX, sizeY, slide_step, patch_base_name(1:end-4))
            [xx, yy] = find(mask ~= 0);
        end
    end
    finish = toc(start);
    disp([patient_id, '   ', num2str(i-2), '/', num2str(length(all_patient_files)-2), 'Time: ', num2str(finish)])
end