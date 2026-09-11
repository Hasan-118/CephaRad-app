import streamlit as st
import trimesh
import pythreejs as p3j
import io

def simplify_stl(file_bytes, target_percent=0.3):
    """
    فشرده‌سازی هوشمند حجم مش جهت افزایش چشمگیر سرعت رندر گرافیکی
    """
    try:
        mesh = trimesh.load(io.BytesIO(file_bytes), file_type='stl')
        # اگر تعداد وجوه مش زیاد باشد، آن را فشرده می‌کند
        if len(mesh.faces) > 30000:
            target_faces = int(len(mesh.faces) * target_percent)
            mesh = mesh.simplify_quadratic_decimation(target_faces)
        return mesh
    except Exception as e:
        st.error(f"خطا در بهینه‌سازی فایل STL: {e}")
        return None

def render_mesh_interactive(mesh, height=450):
    """
    رندر تعاملی سریع با pythreejs
    """
    vertices = mesh.vertices.astype('float32')
    faces = mesh.faces.astype('uint32')
    
    geometry = p3j.BufferGeometry(
        attributes={
            'position': p3j.BufferAttribute(vertices, normalized=False),
            'index': p3j.BufferAttribute(faces.ravel(), normalized=False),
        }
    )
    geometry.exec_three_obj_method('computeVertexNormals')
    
    material = p3j.MeshPhongMaterial(color='#E2E8F0', specular='#111111', shininess=30)
    mesh_3d = p3j.Mesh(geometry=geometry, material=material)
    
    camera = p3j.PerspectiveCamera(position=[0, 0, 80], up=[0, 1, 0])
    light1 = p3j.DirectionalLight(color='white', position=[3, 3, 3], intensity=0.7)
    light2 = p3j.AmbientLight(color='#666666')
    
    scene = p3j.Scene(children=[mesh_3d, camera, light1, light2], background='#0F172A')
    controls = p3j.OrbitControls(controllingCamera=camera)
    renderer = p3j.Renderer(camera=camera, scene=scene, controls=[controls], width=650, height=height)
    
    st.components.v1.html(renderer._repr_html_(), height=height + 20)
