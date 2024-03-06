baseFolder_dir = '\\gel.ulaval.ca\Vision\Usagers\jugir49\Desktop\hdrvdp-3.0.7\hdrvdp-3.0.7\Renders\renders_indoor\';
domaine = "indoor" % "outdoor"

if domaine == "indoor"
    method_list = {'everlight', 'weber22', 'gardner19_3', 'stylelight', 'average_image_texture'};
    image_names =  {"9C4A1948-d05cab6f1f_00.exr", "AG8A6393-4aaa3e4996_00.exr", "AG8A0277-069adeb954_02.exr", "AG8A9235-c076a13635_00.exr", "AG8A7387-28bdfee676_00.exr", "9C4A3235-5e5a966c6a_00.exr", "AG8A9746-14547b1a7d_00.exr", "9C4A8155-0ab964317a_00.exr", "9C4A2747-1a74e2cc6f_00.exr", "9C4A7409-3f66dfc3af_00.exr", "AG8A3554-d4197d3321_00.exr", "AG8A7884-aa41c65ad7_00.exr", "AG8A9772-928d81b078_00.exr", "AG8A7986-85a8ba1cd9_03.exr", "9C4A7584-f649778f07_00.exr", "AG8A8542-042a892f91_00.exr", "9C4A5021-3169a55eaf_00.exr", "9C4A1511-702551eb64_00.exr", "AG8A8625-e4d0806809_00.exr", "AG8A0929-f8c5aa8dd8_06.exr", "9C4A5291-d47902d9d1_00.exr", "AG8A0544-df05e1bb8b_00.exr", "AG8A3511-7466e465bc_00.exr", "9C4A3824-474dfdf650_00.exr", "9C4A4325-efc3b45a84_00.exr"};

else
    method_list = {'everlight_outdoor', 'jinsong_outdoor', 'average_image_outdoor_texture'};
    image_names = {"mud_road_8k_00.exr", "crystal_falls_8k_00.exr", "00038_00.exr", "piazza_martin_lutero_8k_00.exr", "rhodes_memorial_8k_00.exr", "00310_00.exr", "syferfontein_0d_clear_8k_00.exr", "tears_of_steel_bridge_8k_00.exr", "00444_00.exr", "flower_road_8k_00.exr", "openfootage_00025_00.exr", "MIT-01_Ref_00.exr", "monks_forest_8k_00.exr", "harties_8k_00.exr", "winter_sky_8k_00.exr", "openfootage_00141_00.exr", "blue_grotto_8k_00.exr", "chapmans_drive_8k_00.exr", "rustig_koppie_8k_00.exr", "00214_00.exr", "openfootage_00097_00.exr", "00354_00.exr", "mealie_road_8k_00.exr", "HDR_112_River_Road_2_Ref_00.exr", "00236_00.exr"};

harvested_data = ones(length(image_names),length(method_list)).*NaN;
ppd = hdrvdp_pix_per_deg( 27, [2560 1440], 0.7 );

type_of_blender_scenes = {'no_bkg_plane', 'bkg_plane'};
material_list = {'diffuse', 'glossy'};

for type_scene_index = 1:length(type_of_blender_scenes)
    type_of_blender_scene = type_of_blender_scenes{type_scene_index};
    for material_index = 1:length(material_list)
        material = material_list{material_index};
        for i = 1:length(image_names)
            image_name = image_names{i};
            disp(['Rendu à image : ',i])
        
            image_gt_path = strcat(baseFolder_dir, '\', type_of_blender_scene, '\', 'gt_indoor', '\', material, '\', image_name);
            image_gt = exrread(image_gt_path);
            image_gt = image_gt * 400; %% Convertir l'image en cd/m^2
        
            for method_index = 1:length(method_list)
                method = method_list{method_index};
                disp(['Rendu à méthode : ',method])
        
                image_pred_path = strcat(baseFolder_dir, '\', type_of_blender_scene, '\', method, '\', material, '\', image_name);
                image_pred = exrread(image_pred_path);
                image_pred = image_pred * 400; %% Convertir l'image en cd/m^2
        
                score = hdrvdp3('side-by-side', image_pred, image_gt, 'rgb-native', ppd, {'use_gpu', false});
                % value = score.Q_JOD
                harvested_data(i, method_index) = score.Q_JOD;
        
            end
        
        end
        save_path = strcat(type_of_blender_scene, '_', material, "_hdrvdp3_values.mat");
        save(save_path, 'harvested_data', "-mat", "-v7")
    end
end
