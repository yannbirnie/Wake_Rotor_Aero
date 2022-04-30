function solve_lifting_line_system_matrix_approach(rotor_wake_system,wind, Omega, rotorradius) {
      // this codes solves a lifting line model of a horizontal axis rotor
      // as inputs, it takes
      //      rotor_wake_system: data structure that contains the geometry of the horseshoe vortex rings,
      //                         and the control points at the blade
      //      wind: unperturbed wind velocity, also known as U_infinity
      //      Omega: rotational velocity of the rotor
      //      rotorradius: the radius of the rotor

      // get controlpoints data structure
      var controlpoints = rotor_wake_system.controlpoints;
      // get horseshoe vortex rings data structure
      var rings = rotor_wake_system.rings;
      //
      // initialize variables that we will use during the calculation
      var velocity_induced =[]; // velocity induced by a horse vortex ring at a control point
      var up = []; var vp = []; var wp = []; // components of the velocity induced by one horseshoe vortex ring
      var u = 0;  var v = 0;  var w = 0; // total velocity induced at one control point
      var radialposition; var azimdir; // radial position of the control point
      var alpha; // angle of attack
      var GammaNew=[]; // new estimate of bound circulation
      var Gamma=[]; // current solution of bound circulation
      for (var i = 0; i < controlpoints.length; i++) { GammaNew.push(0);}; // initialize as zeros
      var vel1; var vmag; var vaxial; var vazim; var temploads; // velocities and loads at controlpoint
      var MatrixU = new Array(); // matrix of induction, for velocity component in x-direction
      var MatrixV = new Array(); // matrix of induction, for velocity component in y-direction
      var MatrixW = new Array(); // matrix of induction, for velocity component in z-direction
      // output variables
      var a_temp = new Array(); // output vector for axial induction
      var aline_temp = new Array();  // output vector for azimuthal induction
      var r_R_temp = new Array();  // output vector for radial position
      var Fnorm_temp = new Array();  // output vector for axial force
      var Ftan_temp = new Array();  // output vector for tangential force
      var Gamma_temp = new Array();  // output vector for circulation

      // the variables below are to setup the maximum number of iterations and convergence criteria
      var Niterations =1200;
      var errorlimit = 0.01;
      var error = 1.0; var refererror;
      var ConvWeight =0.3;

      // initalize and calculate matrices for velocity induced by horseshoe vortex rings
      // two "for cicles", each line varying wind controlpoint "icp", each column varying with
      // horseshoe vortex ring "jring"
      for (var icp= 0; icp < controlpoints.length; icp++) {
        MatrixU[icp] = new Array(); // new line of matrix
        MatrixV[icp] = new Array(); // new line of matrix
        MatrixW[icp] = new Array(); // new line of matrix
        for (var jring = 0; jring < rings.length; jring++) {
          // set ring strenth to unity, to calculate velocity induced by horseshoe vortex ring "jring"
          // at controlpoint "icp"
          rings[jring] = update_Gamma_sinle_ring(rings[jring],1,1);
          velocity_induced = velocity_induced_single_ring(rings[jring], controlpoints[icp].coordinates);
          // add compnent of velocity per unit strength of circulation to induction matrix
          MatrixU[icp][jring] = velocity_induced[0];
          MatrixV[icp][jring] = velocity_induced[1];
          MatrixW[icp][jring] = velocity_induced[2];
        };
      };

      // calculate solution through an iterative process
      for (var  kiter = 0; kiter < Niterations; kiter++) {

        for (var ig = 0; ig < GammaNew.length; ig++) {
          Gamma[ig] = GammaNew[ig]; //update current bound circulation with new estimate
        }

        // calculate velocity, circulation and loads at the controlpoints
        for (var icp= 0; icp < controlpoints.length; icp++) {
          // determine radial position of the controlpoint;
          radialposition = Math.sqrt(math.dot(controlpoints[icp].coordinates, controlpoints[icp].coordinates));
          u=0; v=0; w=0; // initialize velocity
          // multiply icp line of Matrix with vector of circulation Gamma to calculate velocity at controlpoint
          for (var jring = 0; jring < rings.length; jring++) {
            u = u + MatrixU[icp][jring]*Gamma[jring]; // axial component of velocity
            v= v + MatrixV[icp][jring]*Gamma[jring]; // y-component of velocity
            w= w + MatrixW[icp][jring]*Gamma[jring]; // z-component of velocity
          };
          // calculate total perceived velocity
          vrot = math.cross([-Omega, 0 , 0]  , controlpoints[icp].coordinates ); // rotational velocity
          vel1 = [wind[0]+ u + vrot[0], wind[1]+ v + vrot[1] , wind[2]+ w + vrot[2]]; // total perceived velocity at section
          // calculate azimuthal and axial velocity
          azimdir = math.cross([-1/radialposition, 0 , 0]  , controlpoints[icp].coordinates ); // rotational direction
          vazim = math.dot(azimdir , vel1); // azimuthal direction
          vaxial =  math.dot([1, 0, 0] , vel1); // axial velocity
          // calculate loads using blade element theory
          temploads = loadBladeElement(vaxial, vazim, radialposition/rotorradius);
          // new point of new estimate of circulation for the blade section
          GammaNew[icp] = temploads[2];
          // update output vector
          a_temp[icp] =(-(u + vrot[0])/wind[0]);
          aline_temp[icp] =(vazim/(radialposition*Omega)-1);
          r_R_temp[icp] =(radialposition/rotorradius);
          Fnorm_temp[icp] =(temploads[0]);
          Ftan_temp[icp] =(temploads[1]);
          Gamma_temp[icp] =(temploads[2]);
        }; // end loop control points

        // check convergence of solution
        refererror =math.max(math.abs(GammaNew));
        refererror =Math.max(refererror,0.001); // define scale of bound circulation
        error =math.max(math.abs(math.subtract(GammaNew, Gamma))); // difference betweeen iterations
        error= error/refererror; // relative error
        if (error < errorlimit) {
          // if error smaller than limit, stop iteration cycle
          kiter=Niterations;
        }

        // set new estimate of bound circulation
        for (var ig = 0; ig < GammaNew.length; ig++) {
          GammaNew[ig] = (1-ConvWeight)*Gamma[ig] + ConvWeight*GammaNew[ig];
        }
      }; // end iteration loop

      // output results of converged solution
      return({a: a_temp , aline: aline_temp, r_R: r_R_temp, Fnorm: Fnorm_temp, Ftan: Ftan_temp , Gamma: Gamma_temp});
    };
