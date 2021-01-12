//
// Created by dojing on 2021/1/12.
//

#include "Triangle.h"
#include "TrianglePlanes.h"
std::vector<Triangle> ComputeDelaunayTriangulation (std::vector<support_pt> p_support) {

    // input/output structure for triangulation
    struct triangulateio in, out;
    int32_t k;

    // inputs
    in.numberofpoints = p_support.size();
    in.pointlist = (float*)malloc(in.numberofpoints*2*sizeof(float));
    k=0;

    for (int32_t i=0; i<p_support.size(); i++) {
        in.pointlist[k++] = p_support[i].u;
        in.pointlist[k++] = p_support[i].v;
    }
    in.numberofpointattributes = 0;
    in.pointattributelist      = NULL;
    in.pointmarkerlist         = NULL;
    in.numberofsegments        = 0;
    in.numberofholes           = 0;
    in.numberofregions         = 0;
    in.regionlist              = NULL;

    // outputs
    out.pointlist              = NULL;
    out.pointattributelist     = NULL;
    out.pointmarkerlist        = NULL;
    out.trianglelist           = NULL;
    out.triangleattributelist  = NULL;
    out.neighborlist           = NULL;
    out.segmentlist            = NULL;
    out.segmentmarkerlist      = NULL;
    out.edgelist               = NULL;
    out.edgemarkerlist         = NULL;

    // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary markers)
    char parameters[] = "zQB";
    triangulate(parameters, &in, &out, NULL);

    // put resulting triangles into vector tri
    std::vector<Triangle> tri;
    k=0;
    for (int32_t i=0; i<out.numberoftriangles; i++) {
        tri.push_back(Triangle(out.trianglelist[k],out.trianglelist[k+1],out.trianglelist[k+2]));
        k+=3;
    }

    // free memory used for triangulation
    free(in.pointlist);
    free(out.pointlist);
    free(out.trianglelist);

    // return triangles
    return tri;
}
void ComputePlanes (const std::vector<support_pt>& p_support,std::vector<Triangle> &triangles) {

    // init matrices
    cv::Matx<float,3,3> A;
    cv::Vec<float,3> b;
    cv::Vec<float,3> t;
    // for all triangles do
    for (auto & tri : triangles) {

        // get triangle corner indices
        int32_t c1 = tri.c1;
        int32_t c2 = tri.c2;
        int32_t c3 = tri.c3;

        // compute matrix A for linear system of left triangle
        A(0,0) = p_support[c1].u;A(0,1) = p_support[c1].v; A(0,2) = 1;
        A(1,0) = p_support[c2].u;A(1,1) = p_support[c2].v; A(1,2) = 1;
        A(2,0) = p_support[c3].u;A(2,1) = p_support[c3].v; A(2,2) = 1;

        // compute vector b for linear system (containing the disparities)
        b(0) = p_support[c1].d;
        b(1) = p_support[c2].d;
        b(2) = p_support[c3].d;

        t= A.solve(b,cv::DECOMP_LU);

        tri.t1a = t(0);
        tri.t1b = t(1);
        tri.t1c = t(2);

    }
}
void ComputeDeepth(const std::vector<support_pt>& p_support,const std::vector<Triangle>& tri,cv::Mat& deep) {

    int width = deep.cols;
    int height = deep.rows;

    // loop variables
    int32_t c1, c2, c3;
    float plane_a,plane_b,plane_c;

    // for all triangles do
    for (uint32_t i=0; i<tri.size(); i++) {
        // get plane parameters
        plane_a = tri[i].t1a;
        plane_b = tri[i].t1b;
        plane_c = tri[i].t1c;

        // triangle corners
        c1 = tri[i].c1;
        c2 = tri[i].c2;
        c3 = tri[i].c3;

        // sort triangle corners wrt. u (ascending)
        float tri_u[3];
        tri_u[0] = p_support[c1].u;
        tri_u[1] = p_support[c2].u;
        tri_u[2] = p_support[c3].u;
        float tri_v[3] = {static_cast<float>(p_support[c1].v),static_cast<float>(p_support[c2].v),static_cast<float>(p_support[c3].v)};
        for (uint32_t j=0; j<3; j++) {
            for (uint32_t k=0; k<j; k++) {
                if (tri_u[k]>tri_u[j]) {
                    float tri_u_temp = tri_u[j]; tri_u[j] = tri_u[k]; tri_u[k] = tri_u_temp;
                    float tri_v_temp = tri_v[j]; tri_v[j] = tri_v[k]; tri_v[k] = tri_v_temp;
                }
            }
        }

        // rename corners
        float A_u = tri_u[0]; float A_v = tri_v[0];
        float B_u = tri_u[1]; float B_v = tri_v[1];
        float C_u = tri_u[2]; float C_v = tri_v[2];

        // compute straight lines connecting triangle corners
        float AB_a = 0; float AC_a = 0; float BC_a = 0;
        if ((int32_t)(A_u)!=(int32_t)(B_u)) AB_a = (A_v-B_v)/(A_u-B_u);
        if ((int32_t)(A_u)!=(int32_t)(C_u)) AC_a = (A_v-C_v)/(A_u-C_u);
        if ((int32_t)(B_u)!=(int32_t)(C_u)) BC_a = (B_v-C_v)/(B_u-C_u);
        float AB_b = A_v-AB_a*A_u;
        float AC_b = A_v-AC_a*A_u;
        float BC_b = B_v-BC_a*B_u;

        // a plane is only valid if itself and its projection
        // into the other image is not too much slanted
        bool valid = fabs(plane_a)<0.7;
        int subsampling = 1;
        // first part (triangle corner A->B)
        if ((int32_t)(A_u)!=(int32_t)(B_u)) {
            for (int32_t u=std::max((int32_t)A_u,0); u<std::min((int32_t)B_u,width); u++){
                if (u%subsampling==0) {
                    int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
                    int32_t v_2 = (uint32_t)(AB_a*(float)u+AB_b);
                    for (int32_t v=std::min(v_1,v_2); v<std::max(v_1,v_2); v++)
                        if (v%subsampling==0) {
                            float d_plane  = (plane_a*(float)u+plane_b*(float)v+plane_c);
                            deep.at<cv::Vec3b>(v,u) = cv::Vec3b(d_plane,d_plane,d_plane);
                        }
                }
            }
        }

        // second part (triangle corner B->C)
        if ((int32_t)(B_u)!=(int32_t)(C_u)) {
            for (int32_t u=std::max((int32_t)B_u,0); u<std::min((int32_t)C_u,width); u++){
                if (u%subsampling==0) {
                    int32_t v_1 = (uint32_t)(AC_a*(float)u+AC_b);
                    int32_t v_2 = (uint32_t)(BC_a*(float)u+BC_b);
                    for (int32_t v=std::min(v_1,v_2); v<std::max(v_1,v_2); v++)
                        if (v%subsampling==0) {
                            float d_plane  = (plane_a*(float)u+plane_b*(float)v+plane_c);
                            deep.at<cv::Vec3b>(v,u) = cv::Vec3b(d_plane,d_plane,d_plane);
                        }
                }
            }
        }

    }

}
