//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <optional>

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_normals(const std::vector<Eigen::Vector3f> &normals)
{
    auto id = get_next_id();
    nor_buf.emplace(id, normals);

    normal_id = id;

    return {id};
}

// Bresenham's line drawing algorithm
void rst::rasterizer::draw_line(Eigen::Vector3f begin, Eigen::Vector3f end)
{
    auto x1 = begin.x();
    auto y1 = begin.y();
    auto x2 = end.x();
    auto y2 = end.y();

    Eigen::Vector3f line_color = {255, 255, 255};

    int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;

    dx = x2 - x1;
    dy = y2 - y1;
    dx1 = fabs(dx);
    dy1 = fabs(dy);
    px = 2 * dy1 - dx1;
    py = 2 * dx1 - dy1;

    if (dy1 <= dx1)
    {
        if (dx >= 0)
        {
            x = x1;
            y = y1;
            xe = x2;
        }
        else
        {
            x = x2;
            y = y2;
            xe = x1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point, line_color);
        for (i = 0; x < xe; i++)
        {
            x = x + 1;
            if (px < 0)
            {
                px = px + 2 * dy1;
            }
            else
            {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0))
                {
                    y = y + 1;
                }
                else
                {
                    y = y - 1;
                }
                px = px + 2 * (dy1 - dx1);
            }
            //            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point, line_color);
        }
    }
    else
    {
        if (dy >= 0)
        {
            x = x1;
            y = y1;
            ye = y2;
        }
        else
        {
            x = x2;
            y = y2;
            ye = y1;
        }
        Eigen::Vector2i point = Eigen::Vector2i(x, y);
        set_pixel(point, line_color);
        for (i = 0; y < ye; i++)
        {
            y = y + 1;
            if (py <= 0)
            {
                py = py + 2 * dx1;
            }
            else
            {
                if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0))
                {
                    x = x + 1;
                }
                else
                {
                    x = x - 1;
                }
                py = py + 2 * (dx1 - dy1);
            }
            //            delay(0);
            Eigen::Vector2i point = Eigen::Vector2i(x, y);
            set_pixel(point, line_color);
        }
    }
}

auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(int x, int y, const Vector4f *_v)
{
    Vector3f v[3];
    for (int i = 0; i < 3; i++)
        v[i] = {_v[i].x(), _v[i].y(), 1.0};
    Vector3f f0, f1, f2;
    f0 = v[1].cross(v[0]);
    f1 = v[2].cross(v[1]);
    f2 = v[0].cross(v[2]);
    Vector3f p(x, y, 1.);
    if ((p.dot(f0) * f0.dot(v[2]) > 0) && (p.dot(f1) * f1.dot(v[0]) > 0) && (p.dot(f2) * f2.dot(v[1]) > 0))
        return true;
    return false;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f *v)
{
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
    return {c1, c2, c3};
}

void rst::rasterizer::myDraw(std::vector<Triangle *> &TriangleList)
{
    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;
    Eigen::Matrix4f mvp = projection * view * model;
    for (const auto &t : TriangleList)
    {
        Triangle newtri = *t;
        std::cout << "before draw() transform" << std::endl;
        std::cout << t->a() << std::endl;
        std::array<Eigen::Vector4f, 3> mm{
            (view * model * t->v[0]),
            (view * model * t->v[1]),
            (view * model * t->v[2])};

        std::array<Eigen::Vector3f, 3> viewspace_pos;

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto &v)
                       { return v.template head<3>(); });

        Eigen::Vector4f v[] = {
            mvp * t->v[0],
            mvp * t->v[1],
            mvp * t->v[2]};
        // Homogeneous division
        for (auto &vec : v)
        {
            vec.x() /= vec.w();
            vec.y() /= vec.w();
            vec.z() /= vec.w();
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose();
        Eigen::Vector4f n[] = {
            inv_trans * to_vec4(t->normal[0], 0.0f),
            inv_trans * to_vec4(t->normal[1], 0.0f),
            inv_trans * to_vec4(t->normal[2], 0.0f)};

        // Viewport transformation
        for (auto &vert : v)
        {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            // screen space coordinates
            newtri.setVertex(i, v[i]);
        }

        for (int i = 0; i < 3; ++i)
        {
            // view space normal
            newtri.setNormal(i, n[i].head<3>());
        }

        newtri.setColor(0, 148, 121.0, 92.0);
        newtri.setColor(1, 148, 121.0, 92.0);
        newtri.setColor(2, 148, 121.0, 92.0);
        // Also pass view space vertice position

        rasterize_triangle(newtri, viewspace_pos);
        std::cout << "after draw() transform" << std::endl;
        std::cout << newtri.a() << std::endl;
    }
}

void rst::rasterizer::draw(std::vector<Triangle *> &TriangleList)
{
    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;
    Eigen::Matrix4f mvp = projection * view * model;
    for (const auto &t : TriangleList)
    {
        Triangle newtri = *t;
        std::cout << "before draw() transform" << std::endl;
        std::cout << t->a() << std::endl;
        std::array<Eigen::Vector4f, 3> mm{
            (view * model * t->v[0]),
            (view * model * t->v[1]),
            (view * model * t->v[2])};

        std::array<Eigen::Vector3f, 3> viewspace_pos;

        std::transform(mm.begin(), mm.end(), viewspace_pos.begin(), [](auto &v)
                       { return v.template head<3>(); });

        Eigen::Vector4f v[] = {
            mvp * t->v[0],
            mvp * t->v[1],
            mvp * t->v[2]};
        // Homogeneous division
        for (auto &vec : v)
        {
            vec.x() /= vec.w();
            vec.y() /= vec.w();
            vec.z() /= vec.w();
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose();
        Eigen::Vector4f n[] = {
            inv_trans * to_vec4(t->normal[0], 0.0f),
            inv_trans * to_vec4(t->normal[1], 0.0f),
            inv_trans * to_vec4(t->normal[2], 0.0f)};

        // Viewport transformation
        for (auto &vert : v)
        {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            // screen space coordinates
            newtri.setVertex(i, v[i]);
        }

        for (int i = 0; i < 3; ++i)
        {
            // view space normal
            newtri.setNormal(i, n[i].head<3>());
        }

        newtri.setColor(0, 148, 121.0, 92.0);
        newtri.setColor(1, 148, 121.0, 92.0);
        newtri.setColor(2, 148, 121.0, 92.0);

        // Also pass view space vertice position

        rasterize_triangle(newtri, viewspace_pos);
        std::cout << "after draw() transform" << std::endl;
        std::cout << newtri.a() << std::endl;
    }
}

static Eigen::Vector3f interpolate(float alpha, float beta, float gamma, const Eigen::Vector3f &vert1, const Eigen::Vector3f &vert2, const Eigen::Vector3f &vert3, float weight)
{
    return (alpha * vert1 + beta * vert2 + gamma * vert3) / weight;
}

static Eigen::Vector2f interpolate(float alpha, float beta, float gamma, const Eigen::Vector2f &vert1, const Eigen::Vector2f &vert2, const Eigen::Vector2f &vert3, float weight)
{
    auto u = (alpha * vert1[0] + beta * vert2[0] + gamma * vert3[0]);
    auto v = (alpha * vert1[1] + beta * vert2[1] + gamma * vert3[1]);

    u /= weight;
    v /= weight;

    return Eigen::Vector2f(u, v);
}

float my_interpolation(float x, float y, const Eigen::Vector4f &vert0, const Eigen::Vector4f &vert1, const Eigen::Vector4f &vert2, float weight0, float weight1, float weight2)
{
    float x0 = vert0.x();
    float y0 = vert0.y();

    float x1 = vert1.x();
    float y1 = vert1.y();

    float x2 = vert2.x();
    float y2 = vert2.y();

    float my_z_interpolation = (x - x0) * (y1 - y0) * (weight2 - weight0) +
                               (y - y0) * (weight1 - weight0) * (x2 - x0) - (x1 - x0) * (y - y0) * (weight2 - weight0) - (x - x0) * (y2 - y0) * (weight1 - weight0);
    my_z_interpolation /= ((x2 - x0) * (y1 - y0) - (x1 - x0) * (y2 - y0));
    my_z_interpolation += weight0;
    return my_z_interpolation;
}
// Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle &t, const std::array<Eigen::Vector3f, 3> &view_pos)
{
    // TODO: From your HW3, get the triangle rasterization code.
    // TODO: Inside your rasterization loop:
    //    * v[i].w() is the vertex view space depth value z.
    //    * Z is interpolated view space depth for the current pixel
    //    * zp is depth between zNear and zFar, used for z-buffer

    // float Z = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    // float zp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    // zp *= Z;

    // TODO: Interpolate the attributes:
    // auto interpolated_color
    // auto interpolated_normal
    // auto interpolated_texcoords
    // auto interpolated_shadingcoords

    // Use: fragment_shader_payload payload( interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
    // Use: payload.view_pos = interpolated_shadingcoords;
    // Use: Instead of passing the triangle's color directly to the frame buffer, pass the color to the shaders first to get the final color;
    // Use: auto pixel_color = fragment_shader(payload);
    // std::cout << t.v[0] << std::endl;
    // std::cout << t.v[1] << std::endl;
    // std::cout << t.v[2] << std::endl;
    // std::cout << std::endl;

    auto v = t.toVector4();
    int l = INT_MAX;
    int r = INT_MIN;
    int top = INT_MIN;
    int b = INT_MAX;

    for (auto k : v)
    {
        l = k.x() < l ? k.x() : l;
        r = k.x() > r ? k.x() : r;
        top = k.y() > top ? k.y() : top;
        b = k.y() < b ? k.y() : b;

        // std::cout << "l=" << l << "r=" << r << "top=" << top << "b=" << b << std::endl;
        // std::cout << "kx=" << k.x() << "ky" << k.y() << std::endl;
    }

    for (int i = (int)l; i <= r; i++)
    {
        for (int j = (int)b; j <= top; j++)
        {
            if (insideTriangle(i, j, t.v))
            {
                // std::cout << "reach i=" << i << " j=" << j << std::endl;
                float z_interpolation = my_interpolation(i, j, t.a(), t.b(), t.c(), t.a().z() / t.a().w(), t.b().z() / t.b().w(), t.c().z() / t.c().w()); // answer
                // float z_interpolation = my_interpolation(i, j, t.a(), t.b(), t.c(), t.a().z(), t.b().z(), t.c().z());
                // float z_interpolation = my_interpolation(i, j, t.a(), t.b(), t.c(), t.a().w(), t.b().w(), t.c().w());

                float color_r = my_interpolation(i, j, t.a(), t.b(), t.c(), t.color[0].x(), t.color[1].x(), t.color[2].x());
                float color_g = my_interpolation(i, j, t.a(), t.b(), t.c(), t.color[0].y(), t.color[1].y(), t.color[2].y());
                float color_b = my_interpolation(i, j, t.a(), t.b(), t.c(), t.color[0].z(), t.color[1].z(), t.color[2].z());
                Eigen::Vector3f color_interpolation(color_r, color_g, color_b);

                float nor_x = my_interpolation(i, j, t.a(), t.b(), t.c(), t.normal[0].x(), t.normal[1].x(), t.normal[2].x());
                float nor_y = my_interpolation(i, j, t.a(), t.b(), t.c(), t.normal[0].y(), t.normal[1].y(), t.normal[2].y());
                float nor_z = my_interpolation(i, j, t.a(), t.b(), t.c(), t.normal[0].z(), t.normal[1].z(), t.normal[2].z());
                Eigen::Vector3f nor_interpolation(nor_x, nor_y, nor_z);

                float coord_x = my_interpolation(i, j, t.a(), t.b(), t.c(), t.tex_coords[0].x(), t.tex_coords[1].x(), t.tex_coords[2].x());
                float coord_y = my_interpolation(i, j, t.a(), t.b(), t.c(), t.tex_coords[0].y(), t.tex_coords[1].y(), t.tex_coords[2].y());
                Eigen::Vector2f coords_interpolation(coord_x, coord_y);
                // mipmap1 add
                // mipmap_pixel_uv[i][j] = coords_interpolation;
                // mipmap1 add
                float shading_x = my_interpolation(i, j, t.a(), t.b(), t.c(), view_pos[0].x(), view_pos[1].x(), view_pos[2].x());
                float shading_y = my_interpolation(i, j, t.a(), t.b(), t.c(), view_pos[0].y(), view_pos[1].y(), view_pos[2].y());
                float shading_z = my_interpolation(i, j, t.a(), t.b(), t.c(), view_pos[0].z(), view_pos[1].z(), view_pos[2].z());
                /*
                if (i == 0 || j == 0 || i >= 700 || j >= 700)
                {
                    neighbour = {0, 0, 0, 0}; // won't use anyway, just avoid outrange
                }
                else
                {
                    neighbour << mipmap_pixel_uv[i - 1][j].x(), mipmap_pixel_uv[i - 1][j].y(), mipmap_pixel_uv[i][j - 1].x(), mipmap_pixel_uv[i][j - 1].y();
                }*/

                Eigen::Vector4f neighbour;
                fragment_shader_payload playload(color_interpolation, nor_interpolation, coords_interpolation, neighbour, &texture);
                // fragment_shader_payload playload(color_interpolation, nor_interpolation, coords_interpolation, rst::rasterizer::texture ? &*texture : nullptr);
                playload.view_pos = Eigen::Vector3f(shading_x, shading_y, shading_z);

                auto pixel_color = fragment_shader(playload);

                if (z_interpolation < depth_buf[get_index(i, j)])
                {
                    // The larger depth, the nearer the point is, which should be repainted
                    depth_buf[get_index(i, j)] = z_interpolation;
                    set_pixel(Eigen::Vector2i(i, j), pixel_color);
                    // fragment_shader_payload(const Eigen::Vector3f& col, const Eigen::Vector3f& nor,const Eigen::Vector2f& tc, Texture* tex) :
                    // color(col), normal(nor), tex_coords(tc), texture(tex) {}
                }
            }
        }
    }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f &m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f &v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f &p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
    texture = Texture();
    // texture = std::nullopt;
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height - y) * width + x;
}

void rst::rasterizer::set_pixel(const Vector2i &point, const Eigen::Vector3f &color)
{
    // old index: auto ind = point.y() + point.x() * width;
    int ind = (height - point.y()) * width + point.x();
    frame_buf[ind] = color;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader)
{
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader)
{
    fragment_shader = frag_shader;
}
