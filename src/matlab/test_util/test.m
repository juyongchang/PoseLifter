close all
clear
clc

% 1: Site -> 11
% 2: Neck -> 9
% 3: RightArm -> 15
% 4: RightForeArm -> 16
% 5: RightHand -> 17
% 6: LeftArm -> 12
% 7: LeftForeArm -> 13
% 8: LeftHand -> 14
% 9: RightUpLeg -> 2
% 10: RightLeg -> 3
% 11: RightFoot -> 4
% 12: LeftUpLeg -> 5
% 13: LeftLeg -> 6
% 14: LeftFoot -> 7
% 15: Hip -> 1
% 16: Spine -> 8
% 17: Head -> 10

% Mapping table from H36M to MPI
map = [15, 9, 10, 11, 12, 13, 14, 16, 2, 17, 1, 6, 7, 8, 3, 4, 5];

% Lines
line_idx = {[11 10 9 8 1], ...
            [9 12 13 14], ...
            [9 15 16 17], ...
            [1 2 3 4], ...
            [1 5 6 7]};
line_color = {[0 1 0], ...
              [1 0 0], ...
              [0 0 1], ...
              [0 0 1], ...
              [1 0 0]};

id_sequence = 1;
num_sequence = 6;

annot_name = sprintf('../TS%d/annot_data.mat', id_sequence);
load(annot_name);

idx = find(valid_frame == 1);
num_valid = length(idx);

f = (2048/10)*7.320339203;

id_img = idx(1);
img_name = sprintf('../TS%d/imageSequence/img_%06d.jpg', id_sequence, id_img);
img = imread(img_name);

w = size(img, 2);
h = size(img, 1);
px = w / 2;
py = h / 2;

figure;
h1 = imshow(zeros(h, w, 3, 'uint8')); hold on;
h2 = cell(1,5);
for i = 1:5
    h2{i} = line(zeros(length(line_idx{i}),1), zeros(length(line_idx{i}),1), ...
                 'LineWidth', 4, ...
                 'Color', line_color{i});
end

for i = 1:num_valid
    id_img = idx(i);
    img_name = sprintf('../TS%d/imageSequence/img_%06d.jpg', id_sequence, id_img);
    img = imread(img_name);

    pose3d = annot3(:,map,1,id_img);
    pose2d = annot2(:,map,1,id_img);

    P = [f 0 px; 0 f py; 0 0 1];
    proj = P*pose3d;
    proj = proj ./ repmat(proj(3,:),3,1);

    set(h1, 'CData', img);
    for j = 1:5
        %set(h2{j}, 'XData', squeeze(pose2d(1,line_idx{j})), ...
        %           'YData', squeeze(pose2d(2,line_idx{j})));
        set(h2{j}, 'XData', squeeze(proj(1,line_idx{j})), ...
                   'YData', squeeze(proj(2,line_idx{j})));
    end
    title(sprintf('frame %d/%d', i, num_valid));
    drawnow;
    pause(0.03);
end

