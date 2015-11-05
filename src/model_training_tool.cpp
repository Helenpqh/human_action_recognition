#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/Twist.h>
#include <vector>
#include <fstream>
using namespace std;

const char* file_name="pose_train.dat";
int vector_num=0;

double v_norm(vector<double> &v)
{
  double sum=0.0;
  for(int i=0; i<v.size(); i++){
    sum+=v[i]*v[i];
  }
  return sqrt(sum);
}

vector<double> v_sub(tf::StampedTransform transform1, tf::StampedTransform transform2)
{
  vector<double> v;
  v.push_back(transform1.getOrigin().x()-transform2.getOrigin().x());
  v.push_back(transform1.getOrigin().y()-transform2.getOrigin().y());
  v.push_back(transform1.getOrigin().z()-transform2.getOrigin().z());
  return v;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pose_descriptor_extractor");
  ros::NodeHandle nh_;
  tf::TransformListener listener;
  ros::Rate rate(10.0);
  fstream fout;
  fout.open(file_name, ios::app|ios::out);
  if(!fout.is_open()){
    cerr << "cannnot open file" << endl;
    return false ;
  }

  
  while(nh_.ok())
    {
      tf::StampedTransform transform;
      vector<tf::StampedTransform> transform_v;
      try{
      listener.lookupTransform("/openni_depth_frame", "/head_1",ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame", "/neck_1",ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/torso_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);

      listener.lookupTransform("/openni_depth_frame","/left_shoulder_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame", "/left_elbow_1",ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/left_hand_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);

      listener.lookupTransform("/openni_depth_frame","/right_shoulder_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/right_elbow_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/right_hand_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);

      /*      listener.lookupTransform("/openni_depth_frame","/left_hip_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/left_knee_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/left_foot_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);

      listener.lookupTransform("/openni_depth_frame","/right_hip_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/right_knee_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);
      listener.lookupTransform("/openni_depth_frame","/right_foot_1", ros::Time(0) ,transform);
      transform_v.push_back(transform);*/
      }
      catch (tf::TransformException ex)
        {
          ros::Duration(1.0).sleep();
        }
      if(!transform_v.empty()){
        vector<double> d_l = v_sub(transform_v[5],transform_v[4]);
        vector<double> d_r = v_sub(transform_v[8],transform_v[7]);      
        double ref_dis=max(v_norm(d_l), v_norm(d_r)); //use the ||left/right hand - elbow|| as reference distance
        //        cout<<"distance"<<ref_dis<<endl;
        vector<vector<double> > p(transform_v.size()-1);
        for(int i=0; i<transform_v.size()-1; i++)
          {

            vector<double> temp=(v_sub(transform_v[i+1],transform_v[0])); // every joint relative to head
            for(int j=0; j<temp.size(); j++)
              {
                p[i].push_back(temp[j]/ref_dis);
                fout<< p[i][j]<<",";
              }


          }
            fout<<endl;
            vector_num++;
      }

      rate.sleep();
    }
  cout<<vector_num<<endl;
      fout.close();
  return 0;
}


