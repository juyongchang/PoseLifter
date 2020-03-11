function [] = evaluate_result(opt)
% Evaluate result using PCK3D and AUC measures

% Joint for MPI-3DHP-INF
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

% Drive name
drivename = '/media/juyongchang/5ea9f10d-ae53-447f-96bc-c7002e535930';

% Options
dataset2d = opt.dataset2d;
dataset3d = opt.dataset3d;
canonical = opt.canonical;
mode = 1;
noise = 4;

% Target directory
target_dir = sprintf('%s/2018_pose/PoseLifter/test_inf/resnet152-lift/train2d_%s_train3d_%s/canonical%d_mode%d_noise%d', ...
                     drivename, dataset2d, dataset3d, canonical, mode, noise);

%==========================================================================
% Evaluate code

% Mapping table from H36M to MPI
map = [11, 9, 15, 16, 17, 12, 13, 14, 2, 3, 4, 5, 6, 7, 1, 8, 10];

% Test dataset
test_subject_id = [1,2,3,4,5,6];
test_data_path = '/media/juyongchang/71098543-c79b-4a69-b5fc-949ebbc749a3/Human_Pose_Estimation/MPI_INF_3DHP/test';
data_base_path = [test_data_path filesep 'TS'];

[~,o1,o2,relevant_labels] = mpii_get_joints('relevant');  

% Load results
target_file = sprintf('%s/result.mat', target_dir);
load(target_file);
pred = pred3d;
pred = pred(:,map,:);
pred = pred - pred(:,15,:);
count = 1;

%
sequencewise_per_joint_error = cell(6,1);
sequencewise_activity_labels = cell(6,1);
for i = 1:length(test_subject_id)
    dat = load([data_base_path int2str(test_subject_id(i)) filesep 'annot_data.mat']);
    num_test_points = sum(dat.valid_frame(:));
    per_joint_error = zeros(17,1,num_test_points);
    pje_idx = 1;
    sequencewise_activity_labels{i} = dat.activity_annotation(dat.valid_frame == 1);

    for j = 1:length(dat.valid_frame)
        if(dat.valid_frame(j))
            fprintf('Image %d of %d for Test ID %d\n',j, length(dat.valid_frame), test_subject_id(i));
            error = zeros(17,1);
			
			% The GT has 17 joints, and the order and the annotation of the joints can be observed through the 'relevant_labels' variable
            P = dat.univ_annot3(:,:,:,j)-repmat(dat.univ_annot3(:,15,:,j),1,17);

            % Prediction
			pred_p = squeeze(pred(count,:,:))';

            % Compute error
            count = count + 1;
            error_p = (pred_p - P).^2;
            error_p = sqrt(sum(error_p, 1));
            error(:,1) = error(:,1) + error_p(:);

            % Per joint error
            per_joint_error(:,:,pje_idx) = error;
            pje_idx = pje_idx +1;
        end
    end
    sequencewise_per_joint_error{i} = per_joint_error;
end

% Save evaluation results
save([target_dir filesep 'mpii_3dhp_prediction.mat'], 'sequencewise_per_joint_error', 'sequencewise_activity_labels');
[seq_table, activity_table] = mpii_evaluate_errors(sequencewise_per_joint_error, sequencewise_activity_labels);

out_file = [target_dir filesep 'mpii_3dhp_evaluation'];
writetable(cell2table(seq_table), [out_file '_sequencewise.csv']);
writetable(cell2table(activity_table), [out_file '_activitywise.csv']);

% Sequence-wise PCK3D and AUC
PCK3D = zeros(1,6);
AUC = zeros(1,6);
for i = 1:6
    PCK3D(i) = seq_table{8+i,10};
    AUC(i) = seq_table{15+i,10};
end

% Total PCK3D and AUC
PCK3D_all = activity_table{18,10};
AUC_all = activity_table{27,10};

% Metric for scenes
PCK3D_scene = zeros(1,3);
AUC_scene = zeros(1,3);
PCK3D_scene(1) = (PCK3D(1)*603+PCK3D(2)*540)/(603+540);
PCK3D_scene(2) = (PCK3D(3)*505+PCK3D(4)*553)/(505+553);
PCK3D_scene(3) = (PCK3D(5)*276+PCK3D(6)*452)/(276+452);
AUC_scene(1) = (AUC(1)*603+AUC(2)*540)/(603+540);
AUC_scene(2) = (AUC(3)*505+AUC(4)*553)/(505+553);
AUC_scene(3) = (AUC(5)*276+AUC(6)*452)/(276+452);

% Print
fid = fopen([target_dir filesep 'mpii_result.txt'], 'w');
fprintf(fid, 'PCK3D_all: %.2f\n', PCK3D_all);
fprintf(fid, 'PCK3D_scene: %.2f, %.2f, %.2f\n', PCK3D_scene);
fprintf(fid, 'AUC_all: %.2f\n', AUC_all);
fprintf(fid, 'AUC_scene: %.2f, %.2f, %.2f\n', AUC_scene);
fclose(fid);

