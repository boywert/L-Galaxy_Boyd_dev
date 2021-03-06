#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "allvars.h"
#include "proto.h"


/**@file recipe_infall.c
 * @brief recipe_infall.c calculates the amount of gas that infalls
 *       into the galaxy hot gas component at each time step. This
 *       is derived from the baryonic fraction taking reionization
 *       into account.
 *
 *         */

double infall_recipe(int centralgal, int ngal, double Zcurr)
{
  int i;
  double tot_mass, reionization_modifier, infallingMass;
  double dis;
  double LGAL_TOLERANCE;
  float m1  = 0.01*Hubble_h;
  float m2 = 0.1*Hubble_h;
  int cell;
  //double New_BaryonFrac;
  //double Min_fb;

  /*  need to add up all the baryonic mass asociated with the full halo to check
   *  what baryonic fraction is missing/in excess. That will give the mass of gas
   *  that need to be added/subtracted to the hot phase, gas that infalled.*/
  tot_mass = 0.0;
  LGAL_TOLERANCE = -0.1;//0.000001;
  
  for(i = 0; i < ngal; i++) {    /* Loop over all galaxies in the FoF-halo */

    /* dis is the separation of the galaxy which i orbits from the type 0 */
    dis=separation_gal(centralgal,Gal[i].CentralGal)/(1+ZZ[Halo[Gal[centralgal].HaloNr].SnapNum]);

    /* If galaxy is orbiting a galaxy inside Rvir of the type 0 it will contribute
     * to the baryon sum */
    if ( dis < Gal[centralgal].Rvir ) {
      tot_mass += Gal[i].DiskMass + Gal[i].BulgeMass + Gal[i].ICM + Gal[i].BlackHoleMass;
      tot_mass += Gal[i].ColdGas + Gal[i].HotGas + Gal[i].EjectedMass + Gal[i].BlackHoleGas;
    }
  }

  /* The logic here seems to be that the infalling mass is supposed to be
   * newly-accreted diffuse gas.  So we take the total mass and subtract all
   * the components that we already know about.  In principle, this could lead
   * to negative infall. */
  /* The reionization modifier is applied to the whole baryon mass, not just the 
   * diffuse component.  It is not obvious that this is the correct thing to do. */
  /* The baryonic fraction is conserved by adding/subtracting the infallingMass
   * calculated here to/from the hot gas of the central galaxy of the FOF
   * This is done in main.c where the infall recipe is called.
   * If ReionizationOn>0, the impact of reonization on the fraction of infalling
   * gas is computed, this is done using the Gnedin formalism with a choice
   * of fitting parameters to the formulas proposed by these authors.
   * There are two options. In both cases reionization has the effect of reducing 
   * the fraction of baryons that collapse into dark matter halos, reducing the
   * amount of infalling gas. */
  if(ReionizationOn == 0) {
    reionization_modifier = 1.0;
    infallingMass = reionization_modifier * BaryonFrac * Gal[centralgal].Mvir - tot_mass;
  }
#if defined(READXFRAC) || defined(WITHRADIATIVETRANSFER)
  else if(ReionizationOn == 3) // Ionize all the time
    {
      if(Gal[centralgal].HaloM_Crit200 < m1)
	reionization_modifier = 0.;
      else if(Gal[centralgal].HaloM_Crit200 >= m1 && Gal[centralgal].HaloM_Crit200 <= m2)
	reionization_modifier = log10(Gal[centralgal].HaloM_Crit200/m1)/log10(m2/m1);
      else
	reionization_modifier = 1.0;
      infallingMass = reionization_modifier * (BaryonFrac * Gal[centralgal].Mvir) - tot_mass;
    }
  else if(ReionizationOn == 7) { // Cut off at m=8
    if(Gal[centralgal].Xfrac3d > 0.5) {
      if(Gal[centralgal].HaloM_Crit200 < 0.01*Hubble_h)
	reionization_modifier = 0.;
      else
	reionization_modifier = 1.;
    }
    else
      reionization_modifier = 1.0;
    infallingMass = reionization_modifier*BaryonFrac*Gal[centralgal].Mvir - tot_mass;
  }
  else if(ReionizationOn == 8) {
    if(Gal[centralgal].Xfrac3d > 0.5) { // Cut off at m=9
      if(Gal[centralgal].HaloM_Crit200 < 0.1*Hubble_h)
	reionization_modifier = 0.;
      else
	reionization_modifier = 1.;
    }
    else
      reionization_modifier = 1.0;
    infallingMass = reionization_modifier*BaryonFrac*Gal[centralgal].Mvir - tot_mass;
  }
  else if(ReionizationOn == 6){
    if(Gal[centralgal].Xfrac3d > 0.5) {
      if(Gal[centralgal].HaloM_Crit200 < m1)
	reionization_modifier = 0.;
      else if(Gal[centralgal].HaloM_Crit200 >= m1 && Gal[centralgal].HaloM_Crit200 <= m2)
	reionization_modifier = log10(Gal[centralgal].HaloM_Crit200/m1)/log10(m2/m1);
      else
	reionization_modifier = 1.0;
    }
    else {
      if(Gal[centralgal].HaloM_Crit200 < m1)
	reionization_modifier = 0.;
      else
	reionization_modifier = 1.0;
    }
    infallingMass = reionization_modifier * (BaryonFrac * Gal[centralgal].Mvir) - tot_mass;
  }
#endif
  else if(ReionizationOn == 1 || ReionizationOn == 2) {
    reionization_modifier = do_reionization(Gal[centralgal].Mvir, Zcurr);
    infallingMass = reionization_modifier * BaryonFrac * Gal[centralgal].Mvir - tot_mass;
  }
  //Lx = pow(10,Gal[centralgal].XrayLum)
  //New_BaryonFrac = 0.148*pow((Gal[centralgal].Mvir/(3)),0.148);

  //Min_fb = min(BaryonFrac,New_BaryonFrac);
  
  if (fabs(infallingMass) > LGAL_TOLERANCE)
    {
      infallingMass = reionization_modifier * BaryonFrac * Gal[centralgal].Mvir - tot_mass;
    }

  else
    {
      infallingMass = reionization_modifier * BaryonFrac * Gal[centralgal].Mvir - tot_mass;
    }
  //infallingMass = reionization_modifier * 0.155 * Gal[centralgal].Mvir - tot_mass;
  //double new_fb;
  //new_fb=0.00625*log10(Gal[centralgal].Mvir*1.e10)+0.06125;
  //new_fb=0.02625*log10(Gal[centralgal].Mvir*1.e10)-0.24;
  //new_fb=0.01375*log10(Gal[centralgal].Mvir*1.e10)-0.05;
  //if(Gal[centralgal].Mvir>50.)
  //    new_fb=0.5*(0.0525*log10(Gal[centralgal].Mvir*1.e10)-0.633);
  //else
  //	new_fb=BaryonFrac;

  //if(Gal[centralgal].Mvir>1.)//50.)
    //new_fb=0.15*pow(Gal[centralgal].Mvir/(3.0e+4),0.15);
      //      new_fb=1.5*pow(Gal[centralgal].Mvir/(3.0e+4),0.15);
  //else
  //	new_fb=BaryonFrac;

  //new_fb=BaryonFrac;
  //if(Gal[centralgal].Mvir>50. && Gal[centralgal].Mvir<100.) new_fb= 0.05;
  //if(Gal[centralgal].Mvir>100. && Gal[centralgal].Mvir<500.) new_fb= 0.1;
  //if(Gal[centralgal].Mvir>500. && Gal[centralgal].Mvir<1000.) new_fb= 0.1;
  //if(Gal[centralgal].Mvir>1000. && Gal[centralgal].Mvir<10000.) new_fb= 0.001;
  //if(Gal[centralgal].Mvir>10000. && Gal[centralgal].Mvir<30000.) new_fb= 0.15;
  //infallingMass = reionization_modifier * new_fb * Gal[centralgal].Mvir - tot_mass;



  return infallingMass;
  
}

double do_reionization(float Mvir, double Zcurr)
{
  double modifier, a, alpha;
  //Gnedin (2000)
  double f_of_a, a_on_a0, a_on_ar, Mfiltering, Mjeans, Mchar, mass_to_use;
  double Tvir, Vchar, omegaZ, xZ, deltacritZ, HubbleZ;
  //Okamoto (2008)
  double x0,x,delta_c,delta_0,tau,Mc;
  int tabindex; 
  double f1, f2;

  modifier=1.;

  if (ReionizationOn == 0) {
    printf("Should not be called with this option\n");
    exit(0);
  }
  else if (ReionizationOn == 1) {
    /** reionization recipie described in Gnedin (2000), using the fitting */
    /*  formulas given by Kravtsov et al (2004) Appendix B, used after Delucia 2004*/

    /*  here are two parameters that Kravtsov et al keep fixed. */
    /*  alpha gives the best fit to the Gnedin data */
    alpha = 6.0;
    Tvir = 1e4;

    /*  calculate the filtering mass */

    a = 1.0 / (1.0 + Zcurr);
    a_on_a0 = a / a0;
    a_on_ar = a / ar;

    if(a <= a0)
      f_of_a = 3.0 * a / ((2.0 * alpha) * (5.0 + 2.0 * alpha)) * pow(a_on_a0, alpha);
    else if((a > a0) && (a < ar))
      f_of_a =
	(3.0 / a) * a0 * a0 * (1.0 / (2.0 + alpha) - 2.0 * pow(a_on_a0, -0.5) / (5.0 + 2.0 * alpha)) +
	a * a / 10.0 - (a0 * a0 / 10.0) * (5.0 - 4.0 * pow(a_on_a0, -0.5));
    else
      f_of_a =
	(3.0 / a) * (a0 * a0 * (1.0 / (2.0 + alpha) - 2.0 * pow(a_on_a0, -0.5) / (5.0 + 2.0 * alpha)) +
		     (ar * ar / 10.0) * (5.0 - 4.0 * pow(a_on_ar, -0.5)) - (a0 * a0 / 10.0) * (5.0 -
											       4.0 *
											       pow(a_on_a0,
												   -0.5)) +
		     a * ar / 3.0 - (ar * ar / 3.0) * (3.0 - 2.0 * pow(a_on_ar, -0.5)));
    
    /*  this is in units of 10^10Msun/h, note mu=0.59 and mu^-1.5 = 2.21 */
    Mjeans = 25.0 * pow(Omega, -0.5) * 2.21;
    Mfiltering = Mjeans * pow(f_of_a, 1.5);
    
    
    /*  calculate the characteristic mass coresponding to a halo temperature of 10^4K */
    Vchar = sqrt(Tvir / 36.0);
    omegaZ = Omega * (pow3(1.0 + Zcurr) / (Omega * pow3(1.0 + Zcurr) + OmegaLambda));
    xZ = omegaZ - 1.0;
    deltacritZ = 18.0 * M_PI * M_PI + 82.0 * xZ - 39.0 * xZ * xZ;
    HubbleZ = Hubble * sqrt(Omega * pow3(1.0 + Zcurr) + OmegaLambda);
    
    Mchar = Vchar * Vchar * Vchar / (G * HubbleZ * sqrt(0.5 * deltacritZ));

    /*  we use the maximum of Mfiltering and Mchar */
    mass_to_use = max(Mfiltering, Mchar);
    modifier = 1.0 / pow3(1.0 + 0.26 * (mass_to_use / Mvir));
  }
  else if (ReionizationOn ==2) {
    /* reionization recipie described in Gnedin (2000), with the fitting
     * from Okamoto et al. 2008 -> Qi(2010)*/

    alpha = 2.0;
    a = 1. / (1 + Zcurr);
    a0 = 1.;

    x = - (1 - Omega) * a * a * a / (Omega + (1-Omega) * a * a * a); 
    delta_c = (178 + 82 *x - 39 * x * x) / (1. + x);
  
    x0 = - (1 - Omega) * a0 * a0 * a0 / (Omega + (1-Omega) * a0 * a0 * a0);
    delta_0 = (178 + 82 *x0 - 39 * x0 * x0) / (1. + x0);

    tau = 0.73 * pow(Zcurr + 1,0.18)* exp(-pow(0.25 * Zcurr, 2.1));
  
    Mc = pow(tau  / (1 + Zcurr),3./2) * sqrt(delta_0 / delta_c);

    /* if use Okamoto et al. 2008*/
  
    find_interpolate_reionization(Zcurr, &tabindex, &f1, &f2);
    Mc = f1*log10(Reion_Mc[tabindex])+f2*log10(Reion_Mc[tabindex+1]);
    Mc = pow(10, Mc-10);
    
    modifier = pow(1 + (pow(2, alpha/3.) -1) * pow(Mc / Mvir, alpha), -3./alpha);
  }
  
  return modifier;

}


/* The gas that infalled is added to the hot gas of the central galaxy. 
 *
 * TODO: Note that the algorithm allows negative infall in order to balance 
 * the total baryon content.  In that case, should we decrement the metals
 * accordingly? - or change algorithm NOT to allow negative infall? */

void add_infall_to_hot(int centralgal, int ngal, double infallingGas) {

  if (InfallRecipe == 0)
    {
      /* Check for negative infall and Don't allow! */
      /*if (infallingGas < 0.) {
      //ROB: (07-03-13): This message will never be seen, as add_infall_to_hot() is called IFF infallingGas > 0.0 in main.c
      printf("\ninfallingGas is negative in add_infall_to_hot\n");
      printf("If you really want to do this then you need to rewrite the code.\n");
      printf("This has not been done because it is not obvious how to handle metals:\n");
      printf("- if allowed to outflow then they will be lost for ever;\n");
      printf("- if not then the metallicity of the hotgas will be artificially increased.\n");
      exit(1);
      }*/
      if (infallingGas > 0.)
	{
      /*  Add the infalling gas to the central galaxy hot component */
	  Gal[centralgal].HotGas += infallingGas;
#ifdef INDIVIDUAL_ELEMENTS
	  //Gal[centralgal].HotGas_elements.H += 0.75*infallingGas*1.0e10;
	  Gal[centralgal].HotGas_elements.H += 0.75*(infallingGas/Hubble_h)*1.0e10;
	  Gal[centralgal].HotGas_elements.He += 0.25*(infallingGas/Hubble_h)*1.0e10;
#endif
	}
    }

  else if (InfallRecipe == 1)
    {
      int i;
      double tot_mass;
      double dis;
      double EjectedMass_inter;
      double HGMass_inter;
      double ZGMass_inter;
      double Zfrac;
      double M_infalltoHot;
      double MassExcess;
      double Mass_diff;
      double ZfracHG;
      EjectedMass_inter = 0.;
      HGMass_inter = 0.;
      ZGMass_inter = 0.;
      Zfrac = 0.;
      tot_mass = 0.;
      MassExcess = 0.;
      M_infalltoHot = 0.;
      Mass_diff = 0.;
      ZfracHG = 0.;

      for(i = 0; i < ngal; i++) {    /* Loop over all galaxies in the FoF-halo */

	/* dis is the separation of the galaxy which i orbits from the type 0 */
	dis=separation_gal(centralgal,Gal[i].CentralGal)/(1+ZZ[Halo[Gal[centralgal].HaloNr].SnapNum]);

	/* If galaxy is orbiting a galaxy inside Rvir of the type 0 it will contribute
	 * to the baryon sum */
	if ( dis < Gal[centralgal].Rvir ) {
	  tot_mass += Gal[i].DiskMass + Gal[i].BulgeMass + Gal[i].ICM + Gal[i].BlackHoleMass;
	  tot_mass += Gal[i].ColdGas + Gal[i].HotGas + Gal[i].EjectedMass + Gal[i].BlackHoleGas;
	}
      }

      if (infallingGas > 0.) {
	Gal[centralgal].ExcessMass = max(Gal[centralgal].ExcessMass,infallingGas);
	if(Gal[centralgal].HotGas > 0.)
	  M_infalltoHot = (Gal[centralgal].HotGas/(Gal[centralgal].HotGas + Gal[centralgal].EjectedMass)) * infallingGas;
	else
	  M_infalltoHot = infallingGas;
	if (Gal[centralgal].EjectedMass > 0.){
	  Gal[centralgal].MetalsEjectedMass += Gal[centralgal].MetalsExcessMass * (infallingGas/Gal[centralgal].ExcessMass);
	  Gal[centralgal].MetalsExcessMass -= Gal[centralgal].MetalsExcessMass * (infallingGas/Gal[centralgal].ExcessMass);
	  Gal[centralgal].EjectedMass += Gal[centralgal].ExcessMass * (infallingGas/Gal[centralgal].ExcessMass);
	      
	  Gal[centralgal].ExcessMass -= Gal[centralgal].ExcessMass * (infallingGas/Gal[centralgal].ExcessMass);
	  Gal[centralgal].HotGas += M_infalltoHot;
	  Gal[centralgal].MetalsHotGas += Gal[centralgal].MetalsEjectedMass * (M_infalltoHot/Gal[centralgal].EjectedMass);//infallingGas); //Gal[centralgal].EjectedMass);
	  Gal[centralgal].MetalsEjectedMass -= Gal[centralgal].MetalsEjectedMass * (M_infalltoHot/Gal[centralgal].EjectedMass);//infallingGas);//Gal[centralgal].EjectedMass);
	  Gal[centralgal].EjectedMass -= M_infalltoHot;//Gal[centralgal].EjectedMass * (M_infalltoHot/Gal[centralgal].EjectedMass);
	}
	else {
	  Gal[centralgal].HotGas += Gal[centralgal].ExcessMass * (infallingGas/Gal[centralgal].ExcessMass);
	  Gal[centralgal].MetalsHotGas += Gal[centralgal].MetalsExcessMass * (infallingGas/Gal[centralgal].ExcessMass);
	  Gal[centralgal].MetalsExcessMass -= Gal[centralgal].MetalsExcessMass * (infallingGas/Gal[centralgal].ExcessMass);
	  Gal[centralgal].ExcessMass -= Gal[centralgal].ExcessMass * (infallingGas/Gal[centralgal].ExcessMass);
	}
	  
	//transfer_gas(Gal[centralgal],"Excess",1.,"Ejected",1.,(infallingGas/Gal[centralgal].ExcessMass));
	//transfer_gas(Gal[centralgal],"Ejected",1.,"Hot",1.,(M_infalltoHot/Gal[centralgal].EjectedMass));
	 
      }
      else if (infallingGas < 0. && tot_mass > 0.) {
	HGMass_inter = - (infallingGas/tot_mass)*Gal[centralgal].HotGas;
	ZGMass_inter = - (infallingGas/tot_mass)*Gal[centralgal].MetalsHotGas;
	Gal[centralgal].EjectedMass += min(Gal[centralgal].HotGas,HGMass_inter); 
	Gal[centralgal].MetalsEjectedMass += min(Gal[centralgal].MetalsHotGas,ZGMass_inter); 
	Gal[centralgal].MetalsHotGas -= min(Gal[centralgal].MetalsHotGas,ZGMass_inter);
	Gal[centralgal].HotGas -= min(Gal[centralgal].HotGas,HGMass_inter);
	  
	Mass_diff = Gal[centralgal].EjectedMass + infallingGas;
	
	if (Mass_diff > 0.) {
	  Gal[centralgal].ExcessMass += min(Gal[centralgal].EjectedMass,(- infallingGas));
	  if (Gal[centralgal].EjectedMass > 0.)
	    Zfrac = - (infallingGas/Gal[centralgal].EjectedMass);
	  else
	    Zfrac = 0.;  
	  Gal[centralgal].MetalsExcessMass += min(Gal[centralgal].MetalsEjectedMass,(Zfrac*Gal[centralgal].MetalsEjectedMass));
	  Gal[centralgal].MetalsEjectedMass -= min(Gal[centralgal].MetalsEjectedMass,(Zfrac*Gal[centralgal].MetalsEjectedMass));
	  Gal[centralgal].EjectedMass -= min(Gal[centralgal].EjectedMass,(- infallingGas));
	}
	else if(Mass_diff < 0.) {
	  Gal[centralgal].EjectedMass += min(Gal[centralgal].HotGas,(- Mass_diff)); 
	  if (Gal[centralgal].MetalsHotGas > 0. && Gal[centralgal].HotGas > 0.0)
	    ZfracHG = - (Mass_diff/Gal[centralgal].HotGas);
	  else
	    Zfrac = 0.;
	  Gal[centralgal].MetalsEjectedMass += min(Gal[centralgal].MetalsHotGas,Gal[centralgal].MetalsHotGas*ZfracHG); 
	  Gal[centralgal].MetalsHotGas -= min(Gal[centralgal].MetalsHotGas,Gal[centralgal].MetalsHotGas*ZfracHG);
	  Gal[centralgal].HotGas -= min(Gal[centralgal].HotGas,(- Mass_diff));

	  Gal[centralgal].ExcessMass += min(Gal[centralgal].EjectedMass,(- infallingGas));
	  if (Gal[centralgal].MetalsEjectedMass > 0.0 )
	    Zfrac = - (Mass_diff/Gal[centralgal].EjectedMass);
	  else
	    Zfrac = 0.;
	 
	  Gal[centralgal].MetalsExcessMass += min(Gal[centralgal].MetalsEjectedMass,(Zfrac*Gal[centralgal].MetalsEjectedMass));
	  Gal[centralgal].MetalsEjectedMass -= min(Gal[centralgal].MetalsEjectedMass,(Zfrac*Gal[centralgal].MetalsEjectedMass));
	  Gal[centralgal].EjectedMass -= min(Gal[centralgal].EjectedMass,(- infallingGas));
	} 
      }
      /* if(Zfrac != Zfrac) */
      /* 	printf("zfrac = %lg\n",Zfrac); */
    }

}
