% Create human3.6m dataset

close all;
clear;
clc;

addpaths;

%--------------------------------------------------------------------------
% PARAMETERS

% Subject (1, 5, 6, 7, 8, 9, 11)
SUBJECT = [1 5 6 7 8 9 11];

% Action (2 ~ 16)
ACTION = 2:16;

% Subaction (1 ~ 2)
SUBACTION = 1:2;

% Camera (1 ~ 4)
CAMERA = 1:4;

% Subsampling rate
subsample = 5;

% Control verbosity
verbose = true;

% Drive
drive = '/media/juyongchang/71098543-c79b-4a69-b5fc-949ebbc749a3/Human_Pose_Estimation';

%--------------------------------------------------------------------------
% MAIN LOOP

% For each subject, action, subaction, and camera..
for subject = SUBJECT
    for action = ACTION
        for subaction = SUBACTION
            for camera = CAMERA
                % Print
                fprintf('Processing subject %d, action %d, subaction %d, camera %d..\n', ...
                        subject, action, subaction, camera);

                % Output dataset directory
                dataset_dir = sprintf('%s/H36M/images/s_%02d_act_%02d_subact_%02d_ca_%02d', ...
                                      drive, subject, action, subaction, camera);
                if ~exist(dataset_dir, 'dir')
                    mkdir(dataset_dir);
                end

                if (subject==11) && (action==2) && (subaction==2) && (camera==1)
                    fprintf('There is an error in subject 11, action 2, subaction 2, and camera 1\n');
                    continue;
                end

                %
                filename = sprintf('%s/matlab_meta_new.mat', dataset_dir);
                if exist(filename, 'file')
                    continue;
                end

                % Set features
                Features{1} = H36MPose3DPositionsFeature();
                Features{1}.Part = 'body'; % Only consider 17 joints
                Features{2} = H36MPose2DPositionsFeature();
                Features{2}.Part = 'body'; % Only consider 17 joints
                Features{3} = H36MPose3DPositionsFeature('Monocular', true);
                Features{4} = H36MPose3DPositionsFeature('Monocular', true);
                Features{4}.Part = 'body'; % Only consider 17 joints

                % Select the data
                Sequence = H36MSequence(subject, action, subaction, camera);

                % Compute features
                F = H36MComputeFeatures(Sequence, Features);

                % Subject
                Subject = Sequence.getSubject();
                pos2dSkel = Subject.get2DPosSkel();

                % Camera (in global coordinate)
                Camera = Sequence.getCamera();

                % Camera (in local coordinate)
                Camera0 = H36MCamera(H36MDataBase.instance(), 0, 1);

                % Number of joints
                num_joint = 17;

                %----------------------------------------------------------
                % Get 2D poses
                pose2d = F{2};
                num_frame = size(pose2d, 1);
                pose2d = reshape(pose2d, num_frame, 2, num_joint);

                %----------------------------------------------------------
                % Get 3D poses
                pose3d = F{4};
                num_frame = size(pose3d, 1);
                pose3d = reshape(pose3d, num_frame, 3, num_joint);

                %----------------------------------------------------------
                % Get global 3D poses
                pose3d_global = F{1};
                num_frame = size(pose3d_global, 1);
                pose3d_global = reshape(pose3d_global, num_frame, 3, num_joint);

                %----------------------------------------------------------
                % Camera parameters
                R = repmat(Camera.R(:)', num_frame, 1);
                T = repmat(Camera.T, num_frame, 1);
                f = repmat(Camera.f, num_frame, 1);
                c = repmat(Camera.c, num_frame, 1);

                %----------------------------------------------------------
                % Get RGB images, bounding boxes, and 2D joints
                if verbose
                    fprintf('  Load RGB video: ');
                end
                feat_rgb = H36MRGBVideoFeature();
                da_rgb = feat_rgb.serializer(Sequence);
                num_frames = Sequence.NumFrames;
                if verbose
                    fprintf('Done!!\n');
                end

                % Mask
                if verbose
                    fprintf('  Load mask video: ');
                end
                feat_mask = H36MMyBGMask();
                da_mask = feat_mask.serializer(Sequence);
                if verbose
                    fprintf('Done!!\n');
                end

                % Images to save
                BBOX = zeros(num_frames, 4);
                PTS = zeros(num_frames, 2, num_joint);

                % For each frame,
                for i = 1:num_frames
                    if mod(i,100) == 1
                        fprintf('.');
                    end

                    % Get data
                    rgb = da_rgb.getFrame(i);
                    mask = da_mask.Buffer{i};

                    %------------------------------------------------------
                    % BOUNDING BOX (code from H36MHogFeature.m)

                    % Get the relevant part of the mask (eliminate small components)
                    % bbox: [x_left, y_up, x_width, y_width]
                    [mask, bbox] = preproc_mask(mask, true);
                    if 0
                        imshow(rgb);
                        show2DPose(Camera0.project(F{3}(i,:)), pos2dSkel);
                        rectangle('Position', [bbox(1) bbox(2) bbox(3) bbox(4)], 'EdgeColor', 'r');
                        keyboard;
                    end

                    % Padding
                    MARGIN = [10 10 10 10];
                    bbox = [bbox(1:2) bbox(3:4)-1];

                    props = regionprops(double(mask), 'BoundingBox');

                    if(isempty(props))
                        bbox(1:4) = [1 2 3 4];
                        keyboard;
                    else
                        % FIXME correct boundingbox
                        for j = 1: length(props)
                            boxes(j,:) = [props(j).BoundingBox(1:2) props(j).BoundingBox(1:2)+props(j).BoundingBox(3:4)];
                        end
                        mins = min(boxes,[],1); maxs = max(boxes,[],1);
                        bb = [mins(1) mins(2) maxs(3)-mins(1) maxs(4)-mins(2)];

                        bbox(1) = bb(2); %ymin
                        bbox(2) = bbox(1) + bb(4); %ymax
                        bbox(3) = bb(1); % xmin
                        bbox(4) = bbox(3) + bb(3); % xmax
                    end
                    bbox = round(bbox);
                    bbox(1) = max(bbox(1) - MARGIN(1), 1);
                    bbox(2) = min(bbox(2) + MARGIN(2), size(mask,1));
                    bbox(3) = max(bbox(3) - MARGIN(3), 1);
                    bbox(4) = min(bbox(4) + MARGIN(4), size(mask,2));

                    % Make bounding box square
                    x_width = bbox(4) - bbox(3);
                    y_width = bbox(2) - bbox(1);
                    if x_width > y_width
                        width = x_width;
                        w = x_width - y_width;
                        w1 = floor(w/2);
                        w2 = ceil(w/2);
                        bbox(1) = bbox(1) - w1;
                        bbox(2) = bbox(2) + w2;
                    elseif y_width > x_width
                        width = y_width;
                        w = y_width - x_width;
                        w1 = floor(w/2);
                        w2 = ceil(w/2);
                        bbox(3) = bbox(3) - w1;
                        bbox(4) = bbox(4) + w2;
                    end

                    if 0
                        if (bbox(1) < 1) || (bbox(2) > size(mask,1)) || (bbox(3) < 1) || (bbox(4) > size(mask,2))
                            keyboard;
                        end
                    end

                    % Draw bounding box
                    if 0
                        % RGB
                        imshow(rgb);

                        % Skeleton
                        show2DPose(Camera0.project(F{3}(i,:)), pos2dSkel);

                        % Box
                        rectangle('Position', [bbox(3) bbox(1) bbox(4)-bbox(3) bbox(2)-bbox(1)], ...
                                  'EdgeColor', 'r');

                        %
                        keyboard;
                    end

                    % Crop image
                    img0 = imcrop(rgb, [bbox(3) bbox(1) width width]);
                    w = size(img0, 2);
                    h = size(img0, 1);

                    % Image resize
                    x = max(1, 2-bbox(3));
                    y = max(1, 2-bbox(1));
                    img = zeros(max(w,h), max(w,h), 3, 'uint8');
                    img(y:(y+h-1), x:(x+w-1), :) = img0;
                    img = imresize(img, [256 256]);

                    % Get bounding box and 2d points
                    bbox = [bbox(3) bbox(1) bbox(4)-bbox(3) bbox(2)-bbox(1)]; % [x y w h]
                    assert(bbox(3) == bbox(4));
                    pts = squeeze(pose2d(i,:,:)); % [x; y]
                    pts = pts - repmat([bbox(1); bbox(2)], 1, num_joint);
                    pts = pts ./ repmat([bbox(3); bbox(4)], 1, num_joint);
                    pts = pts * 255 + ones(2, num_joint);

                    % Draw skeleton
                    if 0
                        line_index = cell(1, 5);
                        line_index{1} = [1 2 3 4];
                        line_index{2} = [1 5 6 7];
                        line_index{3} = [1 8 9 10 11];
                        line_index{4} = [9 12 13 14];
                        line_index{5} = [9 15 16 17];
                        imshow(img);
                        hold on;
                        plot(pts(1,:), pts(2,:), 'rs', 'MarkerFaceColor', 'r');
                        for j = 1:5
                            line(pts(1,line_index{j}), pts(2,line_index{j}), 'Color', 'r');
                        end
                    end

                    %
                    BBOX(i,:) = bbox;
                    PTS(i,:,:) = pts;

                    % Save image
                    if mod(i,subsample) == 1
                        img_name = sprintf('%s/s_%02d_act_%02d_subact_%02d_ca_%02d_%06d.jpg', ...
                                           dataset_dir, subject, action, subaction, camera, i);
                        imwrite(img, img_name);
                    end
                end
                fprintf('\n');

                %
                bbox = BBOX;
                pts = PTS;

                %----------------------------------------------------------
                % Save data

                save(filename, 'pose2d', 'bbox', 'pose3d', 'pose3d_global', 'f', 'c', 'R', 'T');
                fprintf('Done!!\n');
            end
        end
    end
end


