#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "grid_subsampling/grid_subsampling.h"

using namespace tensorflow;

REGISTER_OP("BatchGridSubsamplingFeatures")
    .Input("points: float")
    .Input("features: float")
    .Input("labels: int32")
    .Input("batches: int32")
    .Input("dl: float")
    .Input("max_points: int32")
    .Output("sub_points: float")
    .Output("sub_features: float")
    .Output("sub_labels: int32")
    .Output("sub_batches: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input0_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input0_shape));
        ::tensorflow::shape_inference::ShapeHandle input1_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input1_shape));
        ::tensorflow::shape_inference::ShapeHandle input2_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &input2_shape));
        c->set_output(0, input0_shape);
        c->set_output(1, input1_shape);
        c->set_output(2, input2_shape);
        c->set_output(3, c->input(3));
        return Status::OK();
    });





class BatchGridSubsamplingFeaturesOp : public OpKernel {
    public:
    explicit BatchGridSubsamplingFeaturesOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override
    {

        // Grab the input tensors
        const Tensor& points_tensor = context->input(0);
        const Tensor& features_tensor = context->input(1);
        const Tensor& labels_tensor = context->input(2);
        const Tensor& batches_tensor = context->input(3);
        const Tensor& dl_tensor = context->input(4);
        const Tensor& max_p_tensor = context->input(5);

        // check shapes of input and weights
        const TensorShape& points_shape = points_tensor.shape();
        const TensorShape& features_shape = features_tensor.shape();
        const TensorShape& labels_shape = labels_tensor.shape();
        const TensorShape& batches_shape = batches_tensor.shape();

        // check point tensor is a [N x 3] matrix
        DCHECK_EQ(points_shape.dims(), 2);
        DCHECK_EQ(points_shape.dim_size(1), 3);

        // check ranks of features and labels tensors.
        DCHECK_EQ(features_shape.dims(), 2);
        DCHECK_EQ(labels_shape.dims(), 1);

        // Check points, features and labels have the same first dimension
        DCHECK_EQ(points_shape.dim_size(0), features_shape.dim_size(0));
        DCHECK_EQ(points_shape.dim_size(0), labels_shape.dim_size(0));

        // Check that Batch lengths is a vector
        DCHECK_EQ(batches_shape.dims(), 1);

        // Dimensions
        int N = (int)points_shape.dim_size(0);
        int F = (int)features_shape.dim_size(1);

        // Number of batches
        int Nb = (int)batches_shape.dim_size(0);

        // get the data as std vector of points
        float sampleDl = dl_tensor.flat<float>().data()[0];
        int max_p = max_p_tensor.flat<int>().data()[0];
        vector<PointXYZ> original_points = vector<PointXYZ>((PointXYZ*)points_tensor.flat<float>().data(),
                                                            (PointXYZ*)points_tensor.flat<float>().data() + N);

        // get the features as std vector of floats
        vector<float> original_features = vector<float>((float*)features_tensor.flat<float>().data(),
                                                         (float*)features_tensor.flat<float>().data() + N * F);

        // get the features as std vector of floats
        vector<int> original_classes = vector<int>((int*)labels_tensor.flat<int>().data(),
                                                   (int*)labels_tensor.flat<int>().data() + N);

        // Batches lengths
        vector<int> batches = vector<int>((int*)batches_tensor.flat<int>().data(),
                                          (int*)batches_tensor.flat<int>().data() + Nb);


        // Create result containers
        vector<PointXYZ> subsampled_points;
        vector<float> subsampled_features;
        vector<int> subsampled_classes;
        vector<int> subsampled_batches;



        // Compute results
        batch_grid_subsampling(original_points,
                                 subsampled_points,
                                 original_features,
                                 subsampled_features,
                                 original_classes,
                                 subsampled_classes,
                                 batches,
                                 subsampled_batches,
                                 sampleDl,
                                 max_p);

        // Sub_points output
        // *****************

        // create output shape
        TensorShape sub_points_shape;
        sub_points_shape.AddDim(subsampled_points.size());
        sub_points_shape.AddDim(3);

        TensorShape sub_features_shape;
        sub_features_shape.AddDim(subsampled_points.size());
        sub_features_shape.AddDim(F);

        TensorShape sub_labels_shape;
        sub_labels_shape.AddDim(subsampled_points.size());

        // create output tensor
        Tensor* sub_points_output = NULL;
        Tensor* sub_features_output = NULL;
        Tensor* sub_labels_output = NULL;

        OP_REQUIRES_OK(context, context->allocate_output(0, sub_points_shape, &sub_points_output));
        auto sub_points_tensor = sub_points_output->matrix<float>();

        OP_REQUIRES_OK(context, context->allocate_output(1, sub_features_shape, &sub_features_output));
        auto sub_features_tensor = sub_features_output->matrix<float>();

        OP_REQUIRES_OK(context, context->allocate_output(2, sub_labels_shape, &sub_labels_output));
        auto sub_labels_tensor = sub_labels_output->flat<int>();

        // Fill output tensor
        for (int i = 0; i < subsampled_points.size(); i++)
        {
            // Fill points
            sub_points_tensor(i, 0) = subsampled_points[i].x;
            sub_points_tensor(i, 1) = subsampled_points[i].y;
            sub_points_tensor(i, 2) = subsampled_points[i].z;

            // Fill features
            for (int j = 0; j < F; j++)
                sub_features_tensor(i, j) = subsampled_features[F * i + j];

            // Fill labels
            sub_labels_tensor(i) = subsampled_classes[i];

        }

        // Batch length output
        // *******************

        // create output shape
        TensorShape sub_batches_shape;
        sub_batches_shape.AddDim(subsampled_batches.size());

        // create output tensor
        Tensor* sub_batches_output = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(3, sub_batches_shape, &sub_batches_output));
        auto sub_batches_tensor = sub_batches_output->flat<int>();

        // Fill output tensor
        for (int i = 0; i < subsampled_batches.size(); i++)
            sub_batches_tensor(i) = subsampled_batches[i];

        //cout << original_points.size() << " - " << subsampled_points.size() << endl;
        //cout << original_features.size() << " - " << subsampled_features.size() << endl;
        //cout << original_classes.size() << " - " << subsampled_classes.size() << endl;
        //cout << batches.size() << " - " << subsampled_batches.size() << endl;

    }
};


REGISTER_KERNEL_BUILDER(Name("BatchGridSubsamplingFeatures").Device(DEVICE_CPU), BatchGridSubsamplingFeaturesOp);