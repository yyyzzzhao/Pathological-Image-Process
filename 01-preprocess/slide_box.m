function [] = slide_box(img, mask, sizeX, sizeY, slide_step, patch_base_name)
[h, w] = size(mask);
pad_mask = zeros(h, w);
pad_mask(1:slide_step:h, 1:slide_step:w) = 1;
pad = double(mask) .* pad_mask;

[xx, yy] = find(pad ~= 0);
count = 1;
for n = 1:length(xx)
    pt_x = xx(n);  % top-left
    pt_y = yy(n);
    if (pt_x + sizeX - 1) > h || (pt_y + sizeY - 1) > w
        continue
    end
%     pt_x 
%     pt_y
    patch_mask = mask(pt_x : pt_x+sizeX-1, pt_y : pt_y+sizeY-1);
    if mean(patch_mask) < 0.98
        continue
    end
    patch_img = img(pt_x : pt_x+sizeX-1, pt_y : pt_y+sizeY-1, :);
    imwrite(patch_img, [patch_base_name, '_', num2str(count), '.png'])
    count = count + 1;
end