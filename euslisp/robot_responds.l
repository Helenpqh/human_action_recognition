(require :pr2-interface "package://pr2eus/pr2-interface.l")
(ros::roseus-add-msgs "human_action_recognition")
(ros::load-ros-manifest "human_action_recognition")
(ros::load-ros-manifest "roseus")
(ros::roseus "human_interactive")


(pr2-init t)

(setq *cor* nil)
(unless (boundp '*tfl*)
  (setq *tfl* (instance ros::transform-listeren :init)))

(defun interact()
  (let (label coordinate trans-base)
    (ros::subscribe "/human_action_rec_result" human_action_recognition::activityRecognition
                    #'(lambda(msg)
                        (print (list :subscribe (send msg :label)))
                        (setq label (send msg :label))
                        (setq coordinate (send msg :interactionPoint))
                        ))
    (ros::rate 1)
    (while (null label)
      (ros::ros-info "waiting for msgs...")
      (ros::spin-once)
      (ros::sleep)
      (when label
        (print "received msgs")
        (cond
         ((string= label "wave hand")
          (print "execute wave hand"))
         ((string= label "shake hand")
          (print "execte shake hand")
          (setq trans-base
                (send *tfl* :lookup-transform
                      "/base_link"
                      "/head_mount_kinect_ir_link"
                      (ros::time 0)
                      ))
          (setq trs (send (send trans-base :copy-worldcoords)
                          :translate  (ros::tf-point->pos coordinate)))
          (send trs :draw-on :flush t :size 500 :color #f(1 1 0)))
         ; (setq ll (send *pr2* :link-list (send *pr2* :rarm :end-coords :parent)))
         ; (setq mt (send *pr2* :rarm :end-coords))
         ; (send *pr2* :inverse-kinematcs trs
         ;       :move-target mt
         ;       :link-list ll))
         (t ))
        ;;(setq label nil)
        ))    
    ))