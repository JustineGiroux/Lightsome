import os
import bpy 
import os
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"
# remove all ', , and [ ] from the args
argv = [x.replace("'", "") for x in argv]
argv = [x.replace(",", "") for x in argv]
argv = [x.replace("[", "") for x in argv]
argv = [x.replace("]", "") for x in argv]
for i in range(len(argv)):
    print(i, argv[i])
output_dir = argv[0]
hdr_img_root = argv[1]
input_im_name = argv[2]
blender_file_path = argv[3]
material = argv[4]

file = os.path.join(output_dir, input_im_name)

# open blender file
bpy.ops.wm.open_mainfile(filepath=f"{blender_file_path}")

scene = bpy.context.scene
scene.render.image_settings.file_format = 'OPEN_EXR'

scene.render.filepath = 'ambient_parametric'
bpy.data.images[0].filepath = os.path.join(hdr_img_root, input_im_name)

if material == "diffuse":
    ## Specular
    bpy.data.materials["Material"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = 0.0
    ## Roughness
    bpy.data.materials["Material"].node_tree.nodes["Principled BSDF"].inputs[9].default_value = 1.0

elif material == "glossy":
    ## Specular
    bpy.data.materials["Material"].node_tree.nodes["Principled BSDF"].inputs[7].default_value = 1.0
    ## Roughness
    bpy.data.materials["Material"].node_tree.nodes["Principled BSDF"].inputs[9].default_value = 0.1

bpy.data.objects["Sphere.001"].hide_render = True

scene.render.filepath = file

bpy.ops.render.render(write_still=True)