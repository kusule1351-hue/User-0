;;;; spiral.lisp -- the spiral demo, driven from Common Lisp
;;;;
;;;;   sbcl --noinform --non-interactive --load lisp\spiral.lisp
;;;;
;;;; The same network, data and training schedule as build\tinyai.exe -- but every
;;;; decision (what to train, for how long, which sizes to compare) is made here,
;;;; in Lisp, on plain data.  The numeric work (forward pass, back-propagation,
;;;; SGD update) still runs in C and in AVX2 assembly.

(load (merge-pathnames "tinyai.lisp" (or *load-truename* *default-pathname-defaults*)))
(in-package #:tinyai)

;;; An experiment is just a list.  Because it is data, the ordinary list functions
;;; can copy it, edit it and sweep it; no configuration-file format is needed.
(defparameter *experiment*
  '(:data  (:per-class 200 :turns 1.0 :noise 0.4 :seed 1234)
    :net   (:dims (2 64 64 3) :seed 1236)      ; 1236 = data seed + 2, as in tinyai.exe
    :train (:epochs 300 :lr 0.02 :lr-final 0.05 :batch 1)))

(defun plist-with (plist key value)
  "A copy of PLIST with KEY set to VALUE."
  (let ((copy (copy-list plist)))
    (setf (getf copy key) value)
    copy))

(defun plist-without (plist key)
  "A copy of PLIST with KEY removed."
  (loop for (k v) on plist by #'cddr
        unless (eq k key) append (list k v)))

(defun run-experiment (experiment &key (verbose t) plot)
  "Create the data and network EXPERIMENT describes, train, and return the results
of FIT with :PARAMS (the parameter count) added in front."
  (destructuring-bind (&key data net train) experiment
    (let ((test-data (plist-with data :seed (1+ (getf data :seed 1234)))))
      (with-data ((training (apply #'make-spirals data))
                  (testing  (apply #'make-spirals test-data)))
        (with-net ((network (apply #'make-net (getf net :dims) (plist-without net :dims))))
          (when verbose
            (format t "backend : ~a~%model   : ~{~d~^ -> ~}  (~:d parameters)~%~
                       data    : ~a  and  ~a~%~%"
                    (net-backend network) (net-dims network) (net-params network)
                    training testing))
          (let ((result (apply #'fit network training
                               :test testing
                               :stream (and verbose *standard-output*)
                               train)))
            (when plot
              (plot-regions network testing))
            (list* :params (net-params network) result)))))))

;;; ---------------------------------------------------------------------------

(format t "tinyai from Common Lisp -- SBCL ~a~%" (lisp-implementation-version))

(format t "~%== 1. The assembly kernels, called straight from Lisp ==~%")
(format t "kernel set        : ~a~%" (kernel-name))
(format t "(dot '(1 2 3 4) '(5 6 7 8))        => ~a~%" (dot '(1 2 3 4) '(5 6 7 8)))
(let ((y (make-array 4 :element-type 'single-float :initial-element 1f0)))
  (format t "(axpy 2 '(1 2 3 4) #(1 1 1 1))     => ~a~%" (axpy 2 '(1 2 3 4) y)))

(format t "~%== 2. Train the spiral network ==~%")
(run-experiment *experiment* :plot t)

(format t "~%== 3. A sweep: how wide must the hidden layers be? (120 epochs each) ==~%")
(format t "~8@a ~9@a ~8@a ~8@a ~9@a~%" "hidden" "params" "train%" "test%" "seconds")
(dolist (width '(2 4 8 16 32 64))
  (let* ((experiment (plist-with
                      (plist-with *experiment* :net
                                  (plist-with (getf *experiment* :net) :dims (list 2 width width 3)))
                      :train
                      (plist-with (getf *experiment* :train) :epochs 120)))
         (r (run-experiment experiment :verbose nil)))
    (format t "~8d ~9:d ~8,2f ~8,2f ~9,3f~%"
            width (getf r :params) (getf r :train-accuracy) (getf r :test-accuracy)
            (getf r :seconds))))
