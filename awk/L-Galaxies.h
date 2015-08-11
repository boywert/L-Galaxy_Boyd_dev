struct LGalaxy {
   int Type;
   int HaloIndex;
   int SnapNum;
   float LookBackTimeToSnap;
   float CentralMvir;
   float CentralRvir;
   float Pos[3];
   float Vel[3];
   int Len;
   float Mvir;
   float Rvir;
   float Vvir;
   float Vmax;
   float GasSpin[3];
   float StellarSpin[3];
   float InfallVmax;
   float InfallVmaxPeak;
   int InfallSnap;
   float InfallHotGas;
   float HotRadius;
   float OriMergTime;
   float MergTime;
   float DistanceToCentralGal[3];
   float ColdGas;
   float StellarMass;
   float BulgeMass;
   float DiskMass;
   float HotGas;
   float EjectedMass;
   float BlackHoleMass;
   float ICM;
   float MetalsColdGas;
   float MetalsBulgeMass;
   float MetalsDiskMass;
   float MetalsHotGas;
   float MetalsEjectedMass;
   float MetalsICM;
   float PrimordialAccretionRate;
   float CoolingRate;
   float CoolingRate_beforeAGN;
   float Sfr;
   float SfrBulge;
   float XrayLum;
   float BulgeSize;
   float StellarDiskRadius;
   float GasDiskRadius;
   float CosInclination;
   int DisruptOn;
   int MergeOn;
   float CoolingRadius;
   float QuasarAccretionRate;
   float RadioAccretionRate;
   float Mag[40];
   float MagBulge[40];
   float MagDust[40];
   float MassWeightAge;
   float rbandWeightAge;
   int sfh_ibin;
   int sfh_numbins;
   float sfh_DiskMass[20];
   float sfh_BulgeMass[20];
   float sfh_ICM[20];
   float sfh_MetalsDiskMass[20];
   float sfh_MetalsBulgeMass[20];
   float sfh_MetalsICM[20];
};
struct MoMaFGalaxy {
  long long GalID;
  short snapnum;
     short sfh_ibin;
   float sfh_DiskMass;
   float sfh_BulgeMass;
   float sfh_ICM;
   float sfh_MetalsDiskMass;
   float sfh_MetalsBulgeMass;
   float sfh_MetalsICM;
};
