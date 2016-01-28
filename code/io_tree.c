#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "allvars.h"
#include "proto.h"

#ifdef HDF5_INPUT
#include "hdf5.h"
#endif

#ifdef PARALLEL
#include <mpi.h>
#endif

/**@file io_tree.c
 * @brief Reads in the data from the dark matter simulation merger
 *        trees, creates output files and after calculations frees
 *        the allocated memory.
 *
 * There are three different input files: trees_** - normal tree
 * files; tree_dbids - file containing halo IDs (used if GALAXYTREE
 * ON); tree_aux - file containing the particle data (used if
 * UPDATETYPETWO ON).
 */

/**@brief Reads all the tree files if USE_MEMORY_TO_MINIMIZE_IO ON,
 *        otherwise reads in the headers for trees_** and trees_aux;
 *        Opens output files.
 *
 *  If USE_MEMORY_TO_MINIMIZE_IO ON the first step on this file is to
 *  call load_all_dbids(), load_all_auxdata(), load_all_treedata().
 *  These will read all the tree data for the current file into pointers
 *  instead of doing it once for every tree.
 *
 *  If USE_MEMORY_TO_MINIMIZE_IO OFF the trees are read independently
 *  from the files.
 *
 *  Modified versions of myfread/myfwrite/myfseek are called to either
 *  read individual trees from the files into structures or from the
 *  pointers with all the data for the current file into structures.
 *
 *  For each tree the code reads in the header in trees_**: Ntrees -
 *  number of trees in the current file (int); totNHalos - total number
 *  of halos summed over all the trees in current file (int);
 *  TreeNHalos - number of halos in each tree (Ntrees).
 *
 *  Then output files are opened SA_z**_** - for snapshot output;
 *  SA_galtree_** for GALAXYTREE option and SA_**.** for MOMAF.
 *
 *  If UPDATETYPETWO ON the header in tree_aux is also read: NtotHalos -
 *  total number of halos in the file (int); TotIds - total number of
 *  particle IDs (int); Ntrees - total number of trees (int); TotSnaps -
 *  total number of snapshots (int). Define some other quantities:
 *  CountIDs_snaptree - number of Ids for each tree at each snapshot
 *  (TotSnaps * Ntrees); OffsetIDs_snaptree (TotSnaps * Ntrees);
 *  CountIDs_halo - Number of Ids per halo (NtotHalos); OffsetIDs_halo
 *  (int). */

#ifdef HDF5_INPUT
void load_tree_hdf5(int filenr, int *totNHalos) {
  char buf[2048], buf1[200], buf2[200], buf3[200], *memb_name;
  hid_t       file, inttype, floattype, doubletype, float3type, longtype,
    halo_datatype, halo_ids_datatype, space, dset,
    group, dtype, memb_id,native_type, stid;
  hid_t merger_t,attr;
  H5T_class_t  memb_cls, class;
  herr_t      status;
  size_t size;
  hsize_t dims[1] = {0}; 
  hsize_t dim3[1] = {3};
  int i,j,k,errorFlag,ndims,nmembs;

#define HDFFIELDS 300
  void *addr[HDFFIELDS];
  char tag[HDFFIELDS][100];
  int  nt = 0;
  FILE *fd;
  char MergerTree_group_loc[256];
  char NTrees_loc[256];
  char NHalos_loc[256];
  char MergerTree_dataset_loc[256];
  char NHalosInTree_dataset_loc[256];
  char Halo_Data_Descendant[256];
  char Halo_Data_FirstProgenitor[256];
  char Halo_Data_NextProgenitor[256];
  char Halo_Data_FirstHaloInFOFgroup[256];
  char Halo_Data_NextHaloInFOFgroup[256];
  char Halo_Data_Len[256];
  char Halo_Data_M_Mean200[256];
  char Halo_Data_M_Crit200[256];
  char Halo_Data_M_TopHat[256];
  char Halo_Data_Pos[256];
  char Halo_Data_Vel[256];
  char Halo_Data_VelDisp[256];
  char Halo_Data_Vmax[256];
  char Halo_Data_Spin[256];
  char Halo_Data_MostBoundID[256];
  char Halo_Data_SnapNum[256];
  char Halo_Data_FileNr[256];
  char Halo_Data_SubhaloIndex[256];
  char Halo_Data_SubHalfMass[256];
  char HaloIDs_Data_HaloID[256];
  char HaloIDs_Data_FileTreeNr[256];
  char HaloIDs_Data_FirstProgenitor[256];
  char HaloIDs_Data_LastProgenitor[256];
  char HaloIDs_Data_NextProgenitor[256];
  char HaloIDs_Data_Descendant[256];
  char HaloIDs_Data_FirstHaloInFOFgroup[256];
  char HaloIDs_Data_NextHaloInFOFgroup[256];
  char HaloIDs_Data_MainLeafID[256];
  char HaloIDs_Data_Redshift[256];
  char HaloIDs_Data_PeanoKey[256];

  /* define the parameter tags - see HDF5FieldFormatFile */
  strcpy(tag[nt], "MergerTree_group_loc");
  addr[nt] = MergerTree_group_loc;
  nt++;
  strcpy(tag[nt], "NTrees_loc");
  addr[nt] = NTrees_loc;
  nt++;
  strcpy(tag[nt], "NHalos_loc");
  addr[nt] = NHalos_loc;
  nt++;
  strcpy(tag[nt], "MergerTree_dataset_loc");
  addr[nt] = MergerTree_dataset_loc;
  nt++;
  strcpy(tag[nt], "NHalosInTree_dataset_loc");
  addr[nt] = NHalosInTree_dataset_loc;
  nt++;
  strcpy(tag[nt], "Halo_Data_Descendant");
  addr[nt] = Halo_Data_Descendant;
  nt++;
  strcpy(tag[nt], "Halo_Data_FirstProgenitor");
  addr[nt] = Halo_Data_FirstProgenitor;
  nt++;
  strcpy(tag[nt], "Halo_Data_NextProgenitor");
  addr[nt] = Halo_Data_NextProgenitor;
  nt++;
  strcpy(tag[nt], "Halo_Data_FirstHaloInFOFgroup");
  addr[nt] = Halo_Data_FirstHaloInFOFgroup;
  nt++;
  strcpy(tag[nt], "Halo_Data_NextHaloInFOFgroup");
  addr[nt] = Halo_Data_NextHaloInFOFgroup;
  nt++;
  strcpy(tag[nt], "Halo_Data_Len");
  addr[nt] = Halo_Data_Len;
  nt++;
  strcpy(tag[nt], "Halo_Data_M_Mean200");
  addr[nt] = Halo_Data_M_Mean200;
  nt++;
  strcpy(tag[nt], "Halo_Data_M_Crit200");
  addr[nt] = Halo_Data_M_Crit200;
  nt++;
  strcpy(tag[nt], "Halo_Data_M_TopHat");
  addr[nt] = Halo_Data_M_TopHat;
  nt++;
  strcpy(tag[nt], "Halo_Data_Pos");
  addr[nt] = Halo_Data_Pos;
  nt++;
  strcpy(tag[nt], "Halo_Data_Vel");
  addr[nt] = Halo_Data_Vel;
  nt++;
  strcpy(tag[nt], "Halo_Data_VelDisp");
  addr[nt] = Halo_Data_VelDisp;
  nt++;
  strcpy(tag[nt], "Halo_Data_Vmax");
  addr[nt] = Halo_Data_Vmax;
  nt++;
  strcpy(tag[nt], "Halo_Data_Spin");
  addr[nt] = Halo_Data_Spin;
  nt++;
  strcpy(tag[nt], "Halo_Data_MostBoundID");
  addr[nt] = Halo_Data_MostBoundID;
  nt++;
  strcpy(tag[nt], "Halo_Data_SnapNum");
  addr[nt] = Halo_Data_SnapNum;
  nt++;
  strcpy(tag[nt], "Halo_Data_FileNr");
  addr[nt] = Halo_Data_FileNr;
  nt++;
  strcpy(tag[nt], "Halo_Data_SubhaloIndex");
  addr[nt] = Halo_Data_SubhaloIndex;
  nt++;
  strcpy(tag[nt], "Halo_Data_SubHalfMass");
  addr[nt] = Halo_Data_SubHalfMass;
  nt++;
  strcpy(tag[nt], "HaloIDs_Data_HaloID");
  addr[nt] = HaloIDs_Data_HaloID;
  nt++;
  strcpy(tag[nt], "HaloIDs_Data_FileTreeNr");
  addr[nt] = HaloIDs_Data_FileTreeNr;
  nt++;
  strcpy(tag[nt], "HaloIDs_Data_FirstProgenitor");
  addr[nt] = HaloIDs_Data_FirstProgenitor;
  nt++;
  strcpy(tag[nt], "HaloIDs_Data_LastProgenitor");
  addr[nt] = HaloIDs_Data_LastProgenitor;
  nt++;
  strcpy(tag[nt], "HaloIDs_Data_NextProgenitor");
  addr[nt] = HaloIDs_Data_NextProgenitor;
  nt++;
  strcpy(tag[nt], "HaloIDs_Data_Descendant");
  addr[nt] = HaloIDs_Data_Descendant;
  nt++;
  strcpy(tag[nt], "HaloIDs_Data_FirstHaloInFOFgroup");
  addr[nt] = HaloIDs_Data_FirstHaloInFOFgroup;
  nt++;
  strcpy(tag[nt], "HaloIDs_Data_NextHaloInFOFgroup");
  addr[nt] = HaloIDs_Data_NextHaloInFOFgroup;
  nt++;
  strcpy(tag[nt], "HaloIDs_Data_MainLeafID");
  addr[nt] = HaloIDs_Data_MainLeafID;
  nt++;
  strcpy(tag[nt], "HaloIDs_Data_Redshift");
  addr[nt] = HaloIDs_Data_Redshift;
  nt++;
  strcpy(tag[nt], "HaloIDs_Data_PeanoKey");
  addr[nt] = HaloIDs_Data_PeanoKey;
  nt++;
  /* end parameter tags */
  k = 0;
  if((fd = fopen(HDF5_field_file, "r"))) {
    while(!feof(fd)) {
      k++;
      *buf = 0;
      fgets(buf, 2048, fd);
      printf("k = %d\n",k);
      if(sscanf(buf, "%s%s%s", buf1, buf2, buf3) < 2)
	continue;
      if((buf1[0] == '%') | (buf1[0] == '#'))
	continue;
      printf("%s %s %s\n",buf1,buf2,buf3);
      for(i = 0, j = -1; i < nt; i++)
	if(strcmp(buf1, tag[i]) == 0) {
	  j = i;
	  tag[i][0] = 0;
	  break;
	}
      if(j >= 0) {
	strcpy(addr[j], buf2);
	break;
      }
      else {
	printf("Error in file %s: Tag '%s' not allowed or multiple defined.\n", HDF5_field_file, buf1);
	errorFlag = 1;
      }
    }
    fclose(fd);
  }
  else {
    printf("Parameter file %s not found.\n", HDF5_field_file);
    errorFlag = 1;
  }
  for(i = 0; i < nt; i++) {
    if(*tag[i]) {
      printf("Error. I miss a value for tag '%s' in parameter file '%s'.\n", tag[i], HDF5_field_file);
      errorFlag = 1;
    }
  }    
  if(errorFlag)
    terminate("parameterfile incorrect");

  sprintf(buf, "%s/treedata/trees_%d.hdf5", SimulationDir, filenr);
  file = H5Fopen (buf, H5F_ACC_RDONLY, H5P_DEFAULT);
  merger_t = H5Gopen (file, "/MergerTrees", H5P_DEFAULT);
  attr = H5Aopen(merger_t, "NHalos", H5P_DEFAULT);
  status  = H5Aread(attr, H5T_NATIVE_INT, totNHalos);
  H5Aclose(attr);
  attr = H5Aopen(merger_t, "NTrees", H5P_DEFAULT);
  status  = H5Aread(attr, H5T_NATIVE_INT, &Ntrees);
  H5Aclose(attr);
  TreeNHalos = mymalloc("TreeNHalos", sizeof(int) * Ntrees);
  TreeFirstHalo = mymalloc("TreeFirstHalo", sizeof(int) * Ntrees);
  TreeNgals[0] = mymalloc("TreeNgals[n]", NOUT * sizeof(int) * Ntrees);
  dset = H5Dopen (file, "/MergerTrees/Halo", H5P_DEFAULT);
  dtype = H5Dget_type(dset);
  class = H5Tget_class (dtype);
  native_type=H5Tget_native_type(dtype, H5T_DIR_DEFAULT);
  
  if (class == H5T_COMPOUND) {
    nmembs = H5Tget_nmembers(native_type);
    for (i=0; i < nmembs ; i++) {
      memb_id = H5Tget_member_type(native_type, i);
      memb_name = H5Tget_member_name( native_type, i);
      printf("Member: %s\n",memb_name);
      if (H5Tequal (memb_id, H5T_STD_I32LE))
	printf ("  Member %i:  Type is H5T_STD_I32LE\n", i);
      else if (H5Tequal (memb_id, H5T_IEEE_F32LE))
	printf ("  Member %i:  Type is H5T_IEEE_F32LE\n", i);
      else if  (H5Tequal (memb_id, H5T_STD_I64LE))
	printf ("  Member %i:  Type is  H5T_STD_I64LE\n", i);
      else if  (H5Tequal (memb_id, H5T_IEEE_F64LE))
	printf ("  Member %i:  Type is  H5T_IEEE_F64LE\n", i);
      memb_cls = H5Tget_member_class (native_type, i);
      if (memb_cls == H5T_ARRAY) {
	printf ("  Member %i:  Type is  H5T_ARRAY\n", i);
      }
      status = H5Tclose(memb_id);
    }
  }
  
  inttype = H5Tcopy (H5T_STD_I32LE);
  floattype = H5Tcopy (H5T_IEEE_F32LE);
  doubletype = H5Tcopy (H5T_IEEE_F64LE);
  float3type = H5Tarray_create (H5T_IEEE_F32LE, 1, dim3);
  longtype = H5Tcopy (H5T_STD_I64LE);
  
  halo_datatype = H5Tcreate (H5T_COMPOUND, sizeof (struct halo_data));
  status = H5Tinsert (halo_datatype, "Descendant", HOFFSET (struct halo_data, Descendant),
  		      inttype);
  status = H5Tinsert (halo_datatype, "FirstProgenitor", HOFFSET (struct halo_data, FirstProgenitor),
  		      inttype);
  status = H5Tinsert (halo_datatype, "NextProgenitor", HOFFSET (struct halo_data, NextProgenitor),
  		      inttype);
  status = H5Tinsert (halo_datatype, "FirstHaloInFOFgroup", HOFFSET (struct halo_data, FirstHaloInFOFgroup),
  		      inttype);
  status = H5Tinsert (halo_datatype, "NextHaloInFOFgroup", HOFFSET (struct halo_data, NextHaloInFOFgroup),
  		      inttype);
  status = H5Tinsert (halo_datatype, "Len", HOFFSET (struct halo_data, Len),
  		      inttype);
  status = H5Tinsert (halo_datatype, "M_Mean200", HOFFSET (struct halo_data, M_Mean200),
  		      floattype);
  status = H5Tinsert (halo_datatype, "M_Crit200", HOFFSET (struct halo_data, M_Crit200),
  		      floattype);
  status = H5Tinsert (halo_datatype, "M_TopHat", HOFFSET (struct halo_data, M_TopHat),
  		      floattype);
  status = H5Tinsert (halo_datatype, "Pos", HOFFSET (struct halo_data, Pos),
  		      float3type);
  status = H5Tinsert (halo_datatype, "Vel", HOFFSET (struct halo_data, Vel),
  		      float3type);
  status = H5Tinsert (halo_datatype, "VelDisp", HOFFSET (struct halo_data, VelDisp),
  		      floattype);
  status = H5Tinsert (halo_datatype, "Vmax", HOFFSET (struct halo_data, Vmax),
  		      floattype);
  status = H5Tinsert (halo_datatype, "Spin", HOFFSET (struct halo_data, Spin),
  		      float3type);
  status = H5Tinsert (halo_datatype, "MostBoundID", HOFFSET (struct halo_data, MostBoundID),
  		      longtype);
  status = H5Tinsert (halo_datatype, "SnapNum", HOFFSET (struct halo_data, SnapNum),
  		      inttype);
  status = H5Tinsert (halo_datatype, "FileNr", HOFFSET (struct halo_data, FileNr),
  		      inttype);
  status = H5Tinsert (halo_datatype, "SubhaloIndex", HOFFSET (struct halo_data, SubhaloIndex),
  		      inttype);
  status = H5Tinsert (halo_datatype, "SubHalfMass", HOFFSET (struct halo_data, SubHalfMass),
  		      inttype);
  
  space = H5Dget_space (dset);
  ndims = H5Sget_simple_extent_dims (space, dims, NULL);

  Halo_Data = mymalloc("Halo_Data", sizeof(struct halo_data) * (*totNHalos));  
  status = H5Dread (dset, halo_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, Halo_Data);

#ifdef LOADIDS
  halo_ids_datatype = H5Tcreate (H5T_COMPOUND, sizeof (struct halo_ids_data));
  status = H5Tinsert (halo_ids_datatype, "HaloID", HOFFSET (struct halo_ids_data, HaloID),
  		      longtype);
  status = H5Tinsert (halo_ids_datatype, "FileTreeNr", HOFFSET (struct halo_ids_data, FileTreeNr),
  		      longtype);
  status = H5Tinsert (halo_ids_datatype, "FirstProgenitorID", HOFFSET (struct halo_ids_data, FirstProgenitor),
  		      longtype);
  status = H5Tinsert (halo_ids_datatype, "LastProgenitorID", HOFFSET (struct halo_ids_data, LastProgenitor),
  		      longtype);
  status = H5Tinsert (halo_ids_datatype, "NextProgenitorID", HOFFSET (struct halo_ids_data, NextProgenitor),
  		      longtype);
  status = H5Tinsert (halo_ids_datatype, "DescendantID", HOFFSET (struct halo_ids_data, Descendant),
  		      longtype);
  status = H5Tinsert (halo_ids_datatype, "FirstHaloInFOFgroupID", HOFFSET (struct halo_ids_data, FirstHaloInFOFgroup),
  		      longtype);
  status = H5Tinsert (halo_ids_datatype, "NextHaloInFOFgroupID", HOFFSET (struct halo_ids_data, NextHaloInFOFgroup),
  		      longtype);
  status = H5Tinsert (halo_ids_datatype, "Redshift", HOFFSET (struct halo_ids_data, Redshift),
  		      doubletype);
  status = H5Tinsert (halo_ids_datatype, "ChaichalitSrisawat000", HOFFSET (struct halo_ids_data, PeanoKey),
  		      inttype);
  status = H5Tinsert (halo_ids_datatype, "ChaichalitSrisawat001", HOFFSET (struct halo_ids_data, dummy),
  		      inttype);
  HaloIDs_Data = mymalloc("HaloIDs_Data", sizeof(struct halo_ids_data) * (*totNHalos));  
  status = H5Dread (dset, halo_ids_datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, HaloIDs_Data);
#endif
  H5Dclose(dset);
  
  dset = H5Dopen (file, "/MergerTrees/NHalosInTree", H5P_DEFAULT);
  space = H5Dget_space (dset);
  ndims = H5Sget_simple_extent_dims (space, dims, NULL);

  status = H5Dread (dset, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, TreeNHalos);
  H5Dclose(dset);
  H5Fclose(file);
  for(i = 1; i < NOUT; i++)
    TreeNgals[i] = TreeNgals[i - 1] + Ntrees;
  if(Ntrees)
    TreeFirstHalo[0] = 0;
  for(i = 1; i < Ntrees; i++) 
    TreeFirstHalo[i] = TreeFirstHalo[i - 1] + TreeNHalos[i - 1];
#undef HDFFIELDS
}
#endif //HDF5_INPUT
  
void load_tree_table(int filenr) {
  int i,j, n, totNHalos, SnapShotInFileName;
  char buf[1000];
#ifdef READXFRAC
  int cell,status,status_prev;
  double meanxfrac;
#ifdef DP_XFRAC
  double *xfrac;
#else
  float *xfrac;
#endif
#endif

#ifdef  UPDATETYPETWO
  load_all_auxdata(filenr);
#endif

#ifdef PARALLEL
#ifndef MCMC
  printf("\nTask %d reading tree file %d\n", ThisTask, filenr);
#else
  //if def MCMC and PARALLEL only task 0 reads the representative treefile, then broadcasts
  if(ThisTask==0)
    {
      printf("\nTask %d reading trees_%d\n", ThisTask, filenr);
#endif
#endif

      SnapShotInFileName=LastDarkMatterSnapShot;

#ifdef MCMC
#ifdef MR_PLUS_MRII
      SnapShotInFileName=LastDarkMatterSnapShot_MRII;
#endif
#endif

#ifdef LOADIDS
#ifndef MRII
      sprintf(buf, "%s/treedata/tree_dbids_%03d.%d", SimulationDir, SnapShotInFileName, filenr);
#else
      sprintf(buf, "%s/treedata/tree_sf1_dbids_%03d.%d", SimulationDir, SnapShotInFileName, filenr);
#endif
      if(!(treedbids_file = fopen(buf, "r")))
	{
	  char sbuf[1000];
	  sprintf(sbuf, "can't open file `%s'\n", buf);
	  terminate(sbuf);
	}
#endif

#ifndef HDF5_INPUT
      printf("no HDF5_INPUT\n");
#ifndef MRII
      sprintf(buf, "%s/treedata/trees_%03d.%d", SimulationDir, SnapShotInFileName, filenr);
#else
      sprintf(buf, "%s/treedata/trees_sf1_%03d.%d", SimulationDir, SnapShotInFileName, filenr);
#endif


      if(!(tree_file = fopen(buf, "r")))
	{
	  char sbuf[1000];
	  sprintf(sbuf, "can't open file place `%s'\n", buf);
	  terminate(sbuf);
	}

      //read header on trees_** file
      myfread(&Ntrees, 1, sizeof(int), tree_file);
      myfread(&totNHalos, 1, sizeof(int), tree_file);

      TreeNHalos = mymalloc("TreeNHalos", sizeof(int) * Ntrees);
      TreeFirstHalo = mymalloc("TreeFirstHalo", sizeof(int) * Ntrees);
      TreeNgals[0] = mymalloc("TreeNgals[n]", NOUT * sizeof(int) * Ntrees);
      for(n = 1; n < NOUT; n++)
	TreeNgals[n] = TreeNgals[n - 1] + Ntrees;

      myfread(TreeNHalos, Ntrees, sizeof(int), tree_file);

      if(Ntrees)
	TreeFirstHalo[0] = 0;
      for(i = 1; i < Ntrees; i++) {
	TreeFirstHalo[i] = TreeFirstHalo[i - 1] + TreeNHalos[i - 1];
      }

#ifdef PRELOAD_TREES
      Halo_Data = mymalloc("Halo_Data", sizeof(struct halo_data) * totNHalos);
      myfseek(tree_file, sizeof(int) * (2 + Ntrees), SEEK_SET);
      myfread(Halo_Data, totNHalos, sizeof(struct halo_data), tree_file);
      /* for(i=0;i<totNHalos;i++) { */
      /*     printf("ID:%d\n",i); */
      /*     printf("\t Treenr: %d\n",Halo_Data[i].) */
      /*     printf("\t FirstProgenitor: %d\n",Halo_Data[i].FirstProgenitor); */
      /*     printf("\t NextProgenitor: %d\n",Halo_Data[i].NextProgenitor); */
      /*     printf("\t Vmax: %0.8f\n",Halo_Data[i].Vmax); */
      /*     printf("\t M200b: %0.8f\n",Halo_Data[i].M_Mean200); */
      /*     printf("\t M200c: %0.8f\n",Halo_Data[i].M_Crit200); */
      /*     printf("\t M_tophap: %0.8f\n",Halo_Data[i].M_TopHat); */
      /* } */

#ifdef PARALLEL
      printf("\nTask %d done loading trees_%d\n", ThisTask, filenr);
#endif

#ifdef LOADIDS
      HaloIDs_Data = mymalloc("HaloIDs_Data", sizeof(struct halo_ids_data) * totNHalos);
      myfseek(treedbids_file, 0, SEEK_SET);
      myfread(HaloIDs_Data, totNHalos, sizeof(struct halo_ids_data), treedbids_file);
      //for(i=0;i<totNHalos;i++)
      //	printf("id=%lld\n",HaloIDs_Data[i].FirstHaloInFOFgroup);
#ifdef PARALLEL
      printf("\nTask %d done loading tree_dbids_%d\n", ThisTask, filenr);
#endif // PARALLEL

#endif // LOADIDS
#endif // PRELOAD_TREES
#else // HDF5_INPUT
      printf("using HDF5_INPUT\n");
      load_tree_hdf5(filenr, &totNHalos);
#endif // HDF5_INPUT

      //if MCMC is turned only Task 0 reads the file and then broadcasts
#ifdef PARALLEL
#ifdef MCMC
    } // end if ThisTask==0

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Bcast(&Ntrees,sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&totNHalos,sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);

  if(ThisTask>0)
    TreeNHalos = mymalloc("TreeNHalos", sizeof(int) * Ntrees);

  MPI_Bcast(TreeNHalos,sizeof(int)*Ntrees, MPI_BYTE, 0, MPI_COMM_WORLD);

  if(ThisTask>0)
    {
      TreeNgals[0] = mymalloc("TreeNgals[n]", NOUT * sizeof(int) * Ntrees);
      for(n = 1; n < NOUT; n++)
	TreeNgals[n] = TreeNgals[n - 1] + Ntrees;

      TreeFirstHalo = mymalloc("TreeFirstHalo", sizeof(int) * Ntrees);
      if(Ntrees)
	TreeFirstHalo[0] = 0;
      /*Define a variable containing the number you have to jump to
       * get from one firshalo to the next. */
      for(i = 1; i < Ntrees; i++)
	TreeFirstHalo[i] = TreeFirstHalo[i - 1] + TreeNHalos[i - 1];

      Halo_Data = mymalloc("Halo_Data", sizeof(struct halo_data) * totNHalos);
      HaloIDs_Data = mymalloc("HaloIDs_Data", sizeof(struct halo_ids_data) * totNHalos);
    }

  MPI_Bcast(HaloIDs_Data,totNHalos* sizeof(struct halo_ids_data), MPI_BYTE, 0, MPI_COMM_WORLD);

  size_t bytes=totNHalos* sizeof(struct halo_data);
  int ii, Nmessages=10000;
  int HaloChunks=1000000;

  //MPI_BCast has a limit of 2Gb so everything needs to be passed in smaller chunks
  for(ii=0;ii<Nmessages;ii++)
    {
      //if next chunk is outside of array size, just pass whats left and then exit the loop
      if((ii+1)*HaloChunks>totNHalos)
   	{
	  MPI_Bcast(&Halo_Data[ii*HaloChunks],bytes-ii*HaloChunks* sizeof(struct halo_data), MPI_BYTE, 0, MPI_COMM_WORLD);
	  break;
   	}
      else
	MPI_Bcast(&Halo_Data[ii*HaloChunks],HaloChunks* sizeof(struct halo_data), MPI_BYTE, 0, MPI_COMM_WORLD);
    }


  if(ThisTask==0)
    printf("all tree data has now been broadcasted\n");
#endif //MCMC
#endif //PARALLEL

  
#ifdef READXFRAC
  Xfrac_Data = mymalloc("Xfrac_Data", sizeof(float) * totNHalos);
  memset(Xfrac_Data, 10.0, sizeof(float) * totNHalos);
  status_prev=0;
#ifdef PARALLEL
#endif
  if(ThisTask == 0){
    for(i=0;i<NOUT;i++)
      printf("ListOutputSnaps[%d] = %d\n",i,ListOutputSnaps[i]);
  }
  for(i=0;i<ListOutputSnaps[NOUT-1];i++){
#ifdef DP_XFRAC
    xfrac = mymalloc("Xfrac_Read",XfracMesh[0]*XfracMesh[1]*XfracMesh[2]*sizeof(double));
    memset(xfrac, 0.0, sizeof(double)*XfracMesh[0]*XfracMesh[1]*XfracMesh[2]);
#else
    xfrac = mymalloc("Xfrac_Read",XfracMesh[0]*XfracMesh[1]*XfracMesh[2]*sizeof(float));
    memset(xfrac, 0.0, sizeof(float)*XfracMesh[0]*XfracMesh[1]*XfracMesh[2]);
#endif
    if(ThisTask == 0) {
      status = read_xfrac(i,xfrac);
    }
#ifdef PARALLEL
#ifdef DP_XFRAC
    MPI_Bcast(xfrac, XfracMesh[0]*XfracMesh[1]*XfracMesh[2], MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
    MPI_Bcast(xfrac, XfracMesh[0]*XfracMesh[1]*XfracMesh[2], MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

    if(status == 1)  {
      status_prev = 1;
      for(j=0;j<totNHalos;j++) {
	if(Halo_Data[j].SnapNum == i)  {
	  cell = (int) (Halo_Data[j].Pos[0]/(BoxSize/XfracMesh[0]))
	    + (int) (Halo_Data[j].Pos[1]/(BoxSize/XfracMesh[1]))*XfracMesh[0]
	    + (int) (Halo_Data[j].Pos[2]/(BoxSize/XfracMesh[2]))*XfracMesh[0]*XfracMesh[1];
	  Xfrac_Data[j] = xfrac[cell];				
	}
      }
    }
    else {
      for(j=0;j<totNHalos;j++) {
	if(Halo_Data[j].SnapNum == i)  {
	  if(status_prev == 0)
	    Xfrac_Data[j] = 0.;
	  else
	    Xfrac_Data[j] = 1.;
	  // printf("xfrac: %lf\n",Xfrac_Data[j]);
	}
      }
    }
    myfree(xfrac);
  }
#endif 
}




/**@brief Deallocates the arrays used to read in the headers of
 *        trees_** and tree_aux (the ones with more than one
 *        element); if PRELOAD_TREES ON, deallocates
 *        the pointers containing all the input and output data.*/
void free_tree_table(void)
{
#ifdef PRELOAD_TREES
#ifdef READXFRAC
  myfree(Xfrac_Data);
#endif
#ifdef LOADIDS
  myfree(HaloIDs_Data);
#endif
  myfree(Halo_Data);
#endif

  myfree(TreeNgals[0]);

  //deallocates header from trees_**
  myfree(TreeFirstHalo);//derived from the header of trees_**
  myfree(TreeNHalos);

#ifdef UPDATETYPETWO
  myfree(TreeAuxData);
  /*myfree(TreeAuxVel);
    myfree(TreeAuxPos);
    myfree(TreeAuxIds);
    myfree(TreeAuxHeader);*/
#endif
#ifndef HDF5_INPUT
#ifdef LOADIDS
  fclose(treedbids_file);
#endif

  fclose(tree_file);
#endif
}


/**@brief Reads the actual trees into structures to be used in the
 *        code: Halo and HaloIDs; the galaxy structures (HaloGal
 *        and Gal) and the HaloAux are also allocated
 *
 *  If USE_MEMORY_TO_MINIMIZE_IO & NEW_IO are OFF, the trees_** files
 *  are opened every time a tree needs to be read in. Then the code's
 *  structure that will have the tree information is read: Halo =
 *  (sizeof(struct halo_data) * TreeNHalos[]) are read.
 *
 *  HaloAux structure is allocated =
 *  (sizeof(struct halo_aux_data) * TreeNHalos[])
 *
 *  Considering the number of halos allocated for the current tree
 *  the size of the structures containing the galaxies
 *  with a halo and the galaxies with and without a halo to be
 *  allocated is estimated:
 *
 *  For galaxies with a halo - MaxGals = MAXGALFAC * TreeNHalos[] and
 *  HaloGal = (sizeof(struct GALAXY) * MaxGals).
 *
 *  For all galaxies - FoF_MaxGals = 10000*15 and
 *  Gal = (sizeof(struct GALAXY) * FoF_MaxGals)
 *
 *  If GALAXYTREE ON, HaloIDs structure is read from tree_dbids =
 *  sizeof(struct halo_ids_data) * TreeNHalos[] */

void load_tree(int nr)
{
  int i;

#ifdef PRELOAD_TREES
  Halo = Halo_Data + TreeFirstHalo[nr];
  /*for(i=0;i<TreeNHalos[nr];i++)
    printf("vel=%f\n",Halo[i].Vel[1]);*/
#ifdef LOADIDS
  HaloIDs = HaloIDs_Data + TreeFirstHalo[nr];
#endif
#else

  Halo = mymalloc("Halo", sizeof(struct halo_data) * TreeNHalos[nr]);
  myfseek(tree_file, sizeof(int) * (2 + Ntrees) + sizeof(struct halo_data) * TreeFirstHalo[nr], SEEK_SET);
  myfread(Halo, TreeNHalos[nr], sizeof(struct halo_data), tree_file);
#ifdef LOADIDS
  HaloIDs = mymalloc("HaloIDs", sizeof(struct halo_ids_data) * TreeNHalos[nr]);
  myfseek(treedbids_file, sizeof(struct halo_ids_data) * TreeFirstHalo[nr], SEEK_SET);
  myfread(HaloIDs, TreeNHalos[nr], sizeof(struct halo_ids_data), treedbids_file);
#endif

#endif

  //Allocate HaloAux and Galaxy structures.
  HaloAux = mymalloc("HaloAux", sizeof(struct halo_aux_data) * TreeNHalos[nr]);

  for(i = 0; i < TreeNHalos[nr]; i++)
    {
      HaloAux[i].DoneFlag = 0;
      HaloAux[i].HaloFlag = 0;
      HaloAux[i].NGalaxies = 0;
    }

  if(AllocValue_MaxHaloGal == 0)
    AllocValue_MaxHaloGal = 1 + TreeNHalos[nr] / (0.25 * (LastDarkMatterSnapShot+1));

  if(AllocValue_MaxGal == 0)
    AllocValue_MaxGal = 2000;

  MaxHaloGal = AllocValue_MaxHaloGal;
  NHaloGal = 0;
  HaloGal = mymalloc_movable(&HaloGal, "HaloGal", sizeof(struct GALAXY) * MaxHaloGal);
  HaloGalHeap = mymalloc_movable(&HaloGalHeap, "HaloGalHeap", sizeof(int) * MaxHaloGal);
  for(i = 0; i < MaxHaloGal; i++)
    HaloGalHeap[i] = i;

  MaxGal = AllocValue_MaxGal;
  Gal = mymalloc_movable(&Gal, "Gal", sizeof(struct GALAXY) * MaxGal);

#ifdef GALAXYTREE
  if(AllocValue_MaxGalTree == 0)
    AllocValue_MaxGalTree = 1.5 * TreeNHalos[nr];

  MaxGalTree = AllocValue_MaxGalTree;
  GalTree = mymalloc_movable(&GalTree, "GalTree", sizeof(struct galaxy_tree_data) * MaxGalTree);
#endif
}



/**@brief Frees all the Halo and Galaxy structures in the code. */
void free_galaxies_and_tree(void)
{
#ifdef GALAXYTREE
  myfree(GalTree);
#endif
  myfree(Gal);
  myfree(HaloGalHeap);
  myfree(HaloGal);
  myfree(HaloAux);

#ifndef PRELOAD_TREES
#ifdef LOADIDS
  myfree(HaloIDs);
#endif
#ifdef READXFRAC
  myfree(Xfrac);
#endif
  myfree(Halo);
#endif
}


/**@brief Reading routine, either from a file into a structure or
 *        from a pointer to a structure.
 *   */
size_t myfread(void *ptr, size_t size, size_t nmemb, FILE * stream)
{
  size_t nread;

  if(size * nmemb > 0)
    {
      if((nread = fread(ptr, size, nmemb, stream)) != nmemb)
	{
	  if(feof(stream))
	    printf("I/O error (fread) has occured: end of file\n");
	  else
	    printf("I/O error (fread) has occured: %s\n", strerror(errno));
	  fflush(stdout);
	  terminate("read error");
	}
    }
  else
    nread = 0;

  return nread;
}

size_t myfwrite(void *ptr, size_t size, size_t nmemb, FILE * stream)
{
  size_t nwritten;

  if(size * nmemb > 0)
    {
      if((nwritten = fwrite(ptr, size, nmemb, stream)) != nmemb)
	{
	  printf("I/O error (fwrite) has occured: %s\n", strerror(errno));
	  fflush(stdout);
	  terminate("write error");
	}
    }
  else
    nwritten = 0;

  return nwritten;
}

int myfseek(FILE * stream, long offset, int whence)
{
  if(fseek(stream, offset, whence))
    {
      printf("I/O error (fseek) has occured: %s\n", strerror(errno));
      fflush(stdout);
      terminate("write error");
    }

  return 0;
}
