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

%
pose2d = [];
pose3d = [];
bbox = [];
cam_f = [];
cam_c = [];

% For each sequence
for k = 1:6
    id_sequence = k;

    % Load annot data
    annot_name = sprintf('../TS%d/annot_data.mat', id_sequence);
    load(annot_name);

    % Select valid frames
    idx = find(valid_frame == 1);
    num_valid = length(idx);

    % Focal lengths
    if k <= 4
        f = (2048/10)*7.320339203;
    else
        f = (1920/10)*8.770747185;
    end

    % Load first image
    id_img = idx(1);
    img_name = sprintf('../TS%d/imageSequence/img_%06d.jpg', id_sequence, id_img);
    img = imread(img_name);

    % Image size & principal points
    w = size(img, 2);
    h = size(img, 1);
    px = w / 2;
    py = h / 2;

    %
    p2d = permute(squeeze(annot2(:,map,1,idx)), [3 2 1]);
    p2d(:,:,1) = p2d(:,:,1) / w * 255.0;
    p2d(:,:,2) = p2d(:,:,2) / h * 255.0;
    pose2d = cat(1, pose2d, p2d);
    pose3d = cat(1, pose3d, permute(squeeze(univ_annot3(:,map,1,idx)), [3 2 1]));
    bbox = cat(1, bbox, repmat([1 1 w h], num_valid, 1));
    cam_f = cat(1, cam_f, repmat([f f], num_valid, 1));
    cam_c = cat(1, cam_c, repmat([px py], num_valid, 1));
end

% Save data
filename = '../inf.mat';
save(filename, 'pose2d', 'pose3d', 'bbox', 'cam_f', 'cam_c');

