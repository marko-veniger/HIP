/*
Copyright (c) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * hipMemPoolImportExport tests

 */

#include <hip_test_common.hh>
#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>
#include <sys/syscall.h>

constexpr hipMemPoolProps kPoolPropsForExport = {
  hipMemAllocationTypePinned,
  hipMemHandleTypePosixFileDescriptor,
  {
    hipMemLocationTypeDevice,
    0
  },
  nullptr,
  {0}
};

constexpr hipMemPoolProps kPoolPropsNonShareable = {
  hipMemAllocationTypePinned,
  hipMemHandleTypeNone,
  {
    hipMemLocationTypeDevice,
    0
  },
  nullptr,
  {0}
};

TEST_CASE("Unit_hipMemPoolImportExport_Positive") {
    int mem_pool_support = 0;
    HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
	mem_pool_support = 1; /* Temporary workaround untill hipDeviceGetAttribute is fixed TODO: REMOVE BEFORE CREATING PR */
    if (!mem_pool_support) {
        SUCCEED("Runtime doesn't support Memory Pool. Skip the test case.");
        return;
    }
	int fd[2];
	int share_handle;
	int i;
	bool test_result = false;

	hipMemPool_t mem_pool = nullptr;
	hipMemPoolPtrExportData exp_data;
	float *A_d;
	int numElements = 8 * 1024;
	float A_h[numElements];
	float A_h_copy[numElements];
	
	for(i = 0; i < numElements; i++)
	{
		A_h[i] = static_cast<float>(i);
	}
	
	HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolPropsForExport));
	HIP_CHECK(hipMemPoolExportToShareableHandle(&share_handle, mem_pool, hipMemHandleTypePosixFileDescriptor, 0));
	
	

	// create pipe descriptors
	REQUIRE(pipe(fd) == 0);

	auto childpid = fork();
	REQUIRE (childpid >= 0);

	if (childpid > 0) {  // Parent
		// writing only, no need for read-descriptor
		REQUIRE(close(fd[0]) == 0);
		// HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolPropsForExport));
		HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&A_d), numElements * sizeof(float), mem_pool, nullptr));
		
		// HIP_CHECK(hipMemPoolExportToShareableHandle(&share_handle, mem_pool, hipMemHandleTypePosixFileDescriptor, 0));
		
		HIP_CHECK(hipStreamSynchronize(nullptr));
		HIP_CHECK(hipMemcpy(A_d, A_h, numElements * sizeof(float), hipMemcpyHostToDevice));
		
		HIP_CHECK(hipMemPoolExportPointer(&exp_data, &A_d));
		
		// Send shareable handles for mempool and pointer over pipe
		//REQUIRE(write(fd[1], &share_handle, sizeof(int)) >= 0);
		REQUIRE(write(fd[1], &exp_data, sizeof(exp_data)) >= 0);

		REQUIRE(close(fd[1]) == 0);
		REQUIRE(wait(NULL) >= 0);

		HIP_CHECK(hipMemPoolDestroy(mem_pool));

	} else if(childpid == 0){  // Child

		int new_handle = -1;

		close(fd[1]);
		
		// Receive shareable handles for mempool and pointer over pipe
		//read(fd[0], &share_handle, sizeof(int));
		read(fd[0], &exp_data, sizeof(exp_data));
		
		close(fd[0]);
		
		//new_handle = syscall(SYS_pidfd_getfd, getppid(), share_handle, 0);
		new_handle = share_handle;

		HIP_CHECK(hipMemPoolImportFromShareableHandle(&mem_pool, &new_handle, hipMemHandleTypePosixFileDescriptor, 0));
		HIP_CHECK(hipMemPoolImportPointer(reinterpret_cast<void**>(&A_d), mem_pool, &exp_data));
		
		HIP_CHECK(hipMemcpy(A_h_copy, A_d, numElements * sizeof(float), hipMemcpyDeviceToHost));
		
		
		test_result = true;
		for(i = 0; i < numElements; i++) {
			if(A_h[i] != A_h_copy[i]) {
				test_result = false;
				break;
			}
		}
		
		if(!test_result) {
			printf("Multiproc Shared mempool Test Failed!\n");
		} else {
			printf("Multiproc Shared mempool Test OK!\n");
		}
		
		exit(0);
	} else { // fork(2) failed
		printf("fork failed\n");
		REQUIRE(false);
	}
}

TEST_CASE("Unit_hipMemPoolImportExportToShareableHandle_Negative") {
/*
	hipMemPool_t mem_pool = nullptr;

	SECTION("Invalid Shareable Handle") {
		HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolPropsForExport));
		HIP_CHECK_ERROR(hipMemPoolExportToShareableHandle(nullptr, mem_pool, hipMemHandleTypePosixFileDescriptor, 0), hipErrorInvalidValue);
		HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(&mem_pool, nullptr, hipMemHandleTypePosixFileDescriptor, 0), hipErrorInvalidValue);
		HIP_CHECK(hipMemPoolDestroy(mem_pool));
	}
	SECTION("Invalid Handle Type") {
		int share_handle;
		HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolPropsForExport));
		HIP_CHECK_ERROR(hipMemPoolExportToShareableHandle(&share_handle, mem_pool, -1, 0), hipErrorInvalidValue);
		HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(&mem_pool, &share_handle, -1, 0), hipErrorInvalidValue);
		HIP_CHECK(hipMemPoolDestroy(mem_pool));
	}
	SECTION("Invalid Flag") {
		int share_handle;
		HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolPropsForExport));
		HIP_CHECK_ERROR(hipMemPoolExportToShareableHandle(&share_handle, mem_pool, hipMemHandleTypePosixFileDescriptor, -1), hipErrorInvalidValue);
		HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(&mem_pool, &share_handle, hipMemHandleTypePosixFileDescriptor, -1), hipErrorInvalidValue);
		HIP_CHECK(hipMemPoolDestroy(mem_pool));
	}
	
	SECTION("Memory pool invalid properties") {
		int share_handle;
		HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolPropsNonShareable));
		HIP_CHECK_ERROR(hipMemPoolExportToShareableHandle(&share_handle, mem_pool, hipMemHandleTypePosixFileDescriptor, 0), hipErrorInvalidValue);
		HIP_CHECK_ERROR(hipMemPoolImportFromShareableHandle(&mem_pool, &share_handle, hipMemHandleTypePosixFileDescriptor, 0), hipErrorInvalidValue);
		HIP_CHECK(hipMemPoolDestroy(mem_pool));
	}
*/
}
#endif