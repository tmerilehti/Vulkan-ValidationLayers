/* Copyright (c) 2015-2019 The Khronos Group Inc.
 * Copyright (c) 2015-2019 Valve Corporation
 * Copyright (c) 2015-2019 LunarG, Inc.
 * Copyright (C) 2015-2019 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Mark Lobodzinski <mark@lunarg.com>
 * Author: Karl Schultz <karl@lunarg.com>
 * Author: Tony Barbour <tony@LunarG.com>
 */

// Allow use of STL min and max functions in Windows
#define NOMINMAX

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <valarray>

#include "vk_loader_platform.h"
#include "vk_dispatch_table_helper.h"
#include "vk_enum_string_helper.h"
#include "chassis.h"
#include "convert_to_renderpass2.h"
#include "core_validation.h"
#include "shader_validation.h"
#include "vk_layer_utils.h"

// These functions are defined *outside* the core_validation namespace as their type
// is also defined outside that namespace
size_t PipelineLayoutCompatDef::hash() const {
    hash_util::HashCombiner hc;
    // The set number is integral to the CompatDef's distinctiveness
    hc << set << push_constant_ranges.get();
    const auto &descriptor_set_layouts = *set_layouts_id.get();
    for (uint32_t i = 0; i <= set; i++) {
        hc << descriptor_set_layouts[i].get();
    }
    return hc.Value();
}

bool PipelineLayoutCompatDef::operator==(const PipelineLayoutCompatDef &other) const {
    if ((set != other.set) || (push_constant_ranges != other.push_constant_ranges)) {
        return false;
    }

    if (set_layouts_id == other.set_layouts_id) {
        // if it's the same set_layouts_id, then *any* subset will match
        return true;
    }

    // They aren't exactly the same PipelineLayoutSetLayouts, so we need to check if the required subsets match
    const auto &descriptor_set_layouts = *set_layouts_id.get();
    assert(set < descriptor_set_layouts.size());
    const auto &other_ds_layouts = *other.set_layouts_id.get();
    assert(set < other_ds_layouts.size());
    for (uint32_t i = 0; i <= set; i++) {
        if (descriptor_set_layouts[i] != other_ds_layouts[i]) {
            return false;
        }
    }
    return true;
}

using std::max;
using std::string;
using std::stringstream;
using std::unique_ptr;
using std::unordered_map;
using std::unordered_set;
using std::vector;

QUEUE_STATE *CoreChecks::GetQueueState(VkQueue queue) {
    auto it = queueMap.find(queue);
    if (it == queueMap.end()) {
        return nullptr;
    }
    return &it->second;
}

COMMAND_POOL_STATE *CoreChecks::GetCommandPoolState(VkCommandPool pool) {
    auto it = commandPoolMap.find(pool);
    if (it == commandPoolMap.end()) {
        return nullptr;
    }
    return it->second.get();
}

PHYSICAL_DEVICE_STATE *CoreChecks::GetPhysicalDeviceState(VkPhysicalDevice phys) {
    auto *phys_dev_map = ((physical_device_map.size() > 0) ? &physical_device_map : &instance_state->physical_device_map);
    auto it = phys_dev_map->find(phys);
    if (it == phys_dev_map->end()) {
        return nullptr;
    }
    return &it->second;
}

PHYSICAL_DEVICE_STATE *CoreChecks::GetPhysicalDeviceState() { return physical_device_state; }

std::string FormatDebugLabel(const char *prefix, const LoggingLabel &label) {
    if (label.Empty()) return std::string();
    std::string out;
    string_sprintf(&out, "%sVkDebugUtilsLabel(name='%s' color=[%g, %g %g, %g])", prefix, label.name.c_str(), label.color[0],
                   label.color[1], label.color[2], label.color[3]);
    return out;
}

// Retrieve pipeline node ptr for given pipeline object
PIPELINE_STATE *CoreChecks::GetPipelineState(VkPipeline pipeline) {
    auto it = pipelineMap.find(pipeline);
    if (it == pipelineMap.end()) {
        return nullptr;
    }
    return it->second.get();
}

PIPELINE_LAYOUT_STATE const *CoreChecks::GetPipelineLayout(VkPipelineLayout pipeLayout) {
    auto it = pipelineLayoutMap.find(pipeLayout);
    if (it == pipelineLayoutMap.end()) {
        return nullptr;
    }
    return it->second.get();
}

std::shared_ptr<cvdescriptorset::DescriptorSetLayout const> const GetDescriptorSetLayout(CoreChecks const *dev_data,
                                                                                         VkDescriptorSetLayout dsLayout) {
    auto it = dev_data->descriptorSetLayoutMap.find(dsLayout);
    if (it == dev_data->descriptorSetLayoutMap.end()) {
        return nullptr;
    }
    return it->second;
}

SHADER_MODULE_STATE const *CoreChecks::GetShaderModuleState(VkShaderModule module) {
    auto it = shaderModuleMap.find(module);
    if (it == shaderModuleMap.end()) {
        return nullptr;
    }
    return it->second.get();
}

bool CoreChecks::LogInvalidAttachmentMessage(const char *type1_string, const RENDER_PASS_STATE *rp1_state, const char *type2_string,
                                             const RENDER_PASS_STATE *rp2_state, uint32_t primary_attach, uint32_t secondary_attach,
                                             const char *msg, const char *caller, const char *error_code) {
    return log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_RENDER_PASS_EXT,
                   HandleToUint64(rp1_state->renderPass), error_code,
                   "%s: RenderPasses incompatible between %s w/ renderPass %s and %s w/ renderPass %s Attachment %u is not "
                   "compatible with %u: %s.",
                   caller, type1_string, report_data->FormatHandle(rp1_state->renderPass).c_str(), type2_string,
                   report_data->FormatHandle(rp2_state->renderPass).c_str(), primary_attach, secondary_attach, msg);
}

// Return Set node ptr for specified set or else NULL
cvdescriptorset::DescriptorSet *CoreChecks::GetSetNode(VkDescriptorSet set) {
    auto set_it = setMap.find(set);
    if (set_it == setMap.end()) {
        return NULL;
    }
    return set_it->second.get();
}

// Block of code at start here specifically for managing/tracking DSs

// Return Pool node ptr for specified pool or else NULL
DESCRIPTOR_POOL_STATE *CoreChecks::GetDescriptorPoolState(const VkDescriptorPool pool) {
    auto pool_it = descriptorPoolMap.find(pool);
    if (pool_it == descriptorPoolMap.end()) {
        return NULL;
    }
    return pool_it->second.get();
}

// Remove set from setMap and delete the set
void CoreChecks::FreeDescriptorSet(cvdescriptorset::DescriptorSet *descriptor_set) { setMap.erase(descriptor_set->GetSet()); }

// Free all DS Pools including their Sets & related sub-structs
// NOTE : Calls to this function should be wrapped in mutex
void CoreChecks::DeletePools() {
    for (auto ii = descriptorPoolMap.begin(); ii != descriptorPoolMap.end();) {
        // Remove this pools' sets from setMap and delete them
        for (auto ds : ii->second->sets) {
            FreeDescriptorSet(ds);
        }
        ii->second->sets.clear();
        ii = descriptorPoolMap.erase(ii);
    }
}

// For given CB object, fetch associated CB Node from map
CMD_BUFFER_STATE *CoreChecks::GetCBState(const VkCommandBuffer cb) {
    auto it = commandBufferMap.find(cb);
    if (it == commandBufferMap.end()) {
        return NULL;
    }
    return it->second.get();
}

// Tie the VK_OBJECT to the cmd buffer which includes:
//  Add object_binding to cmd buffer
//  Add cb_binding to object
static void AddCommandBufferBinding(std::unordered_set<CMD_BUFFER_STATE *> *cb_bindings, VK_OBJECT obj, CMD_BUFFER_STATE *cb_node) {
    cb_bindings->insert(cb_node);
    cb_node->object_bindings.insert(obj);
}
// Reset the command buffer state
//  Maintain the createInfo and set state to CB_NEW, but clear all other state
void CoreChecks::ResetCommandBufferState(const VkCommandBuffer cb) {
    CMD_BUFFER_STATE *pCB = GetCBState(cb);
    if (pCB) {
        pCB->in_use.store(0);
        // Reset CB state (note that createInfo is not cleared)
        pCB->commandBuffer = cb;
        memset(&pCB->beginInfo, 0, sizeof(VkCommandBufferBeginInfo));
        memset(&pCB->inheritanceInfo, 0, sizeof(VkCommandBufferInheritanceInfo));
        pCB->hasDrawCmd = false;
        pCB->state = CB_NEW;
        pCB->submitCount = 0;
        pCB->image_layout_change_count = 1;  // Start at 1. 0 is insert value for validation cache versions, s.t. new == dirty
        pCB->status = 0;
        pCB->static_status = 0;
        pCB->viewportMask = 0;
        pCB->scissorMask = 0;

        for (auto &item : pCB->lastBound) {
            item.second.reset();
        }

        memset(&pCB->activeRenderPassBeginInfo, 0, sizeof(pCB->activeRenderPassBeginInfo));
        pCB->activeRenderPass = nullptr;
        pCB->activeSubpassContents = VK_SUBPASS_CONTENTS_INLINE;
        pCB->activeSubpass = 0;
        pCB->broken_bindings.clear();
        pCB->waitedEvents.clear();
        pCB->events.clear();
        pCB->writeEventsBeforeWait.clear();
        pCB->activeQueries.clear();
        pCB->startedQueries.clear();
        pCB->eventToStageMap.clear();
        pCB->vertex_buffer_used = false;
        pCB->primaryCommandBuffer = VK_NULL_HANDLE;
        // If secondary, invalidate any primary command buffer that may call us.
        if (pCB->createInfo.level == VK_COMMAND_BUFFER_LEVEL_SECONDARY) {
            InvalidateCommandBuffers(pCB->linkedCommandBuffers, {HandleToUint64(cb), kVulkanObjectTypeCommandBuffer});
        }

        // Remove reverse command buffer links.
        for (auto pSubCB : pCB->linkedCommandBuffers) {
            pSubCB->linkedCommandBuffers.erase(pCB);
        }
        pCB->linkedCommandBuffers.clear();
        pCB->updateImages.clear();
        pCB->updateBuffers.clear();
        pCB->queue_submit_functions.clear();
        pCB->cmd_execute_commands_functions.clear();
        pCB->eventUpdates.clear();
        pCB->queryUpdates.clear();

        pCB->object_bindings.clear();
        pCB->activeFramebuffer = VK_NULL_HANDLE;

        // Clean up the label data
        ResetCmdDebugUtilsLabel(report_data, pCB->commandBuffer);
        pCB->debug_label.Reset();
    }

    if (enabled.gpu_validation) {
        GpuResetCommandBuffer(cb);
    }
}
void CoreChecks::InitGpuValidation() {
    // Process the layer settings file.
    enum CoreValidationGpuFlagBits {
        CORE_VALIDATION_GPU_VALIDATION_ALL_BIT = 0x00000001,
        CORE_VALIDATION_GPU_VALIDATION_RESERVE_BINDING_SLOT_BIT = 0x00000002,
    };
    typedef VkFlags CoreGPUFlags;
    static const std::unordered_map<std::string, VkFlags> gpu_flags_option_definitions = {
        {std::string("all"), CORE_VALIDATION_GPU_VALIDATION_ALL_BIT},
        {std::string("reserve_binding_slot"), CORE_VALIDATION_GPU_VALIDATION_RESERVE_BINDING_SLOT_BIT},
    };
    std::string gpu_flags_key = "lunarg_core_validation.gpu_validation";
    CoreGPUFlags gpu_flags = GetLayerOptionFlags(gpu_flags_key, gpu_flags_option_definitions, 0);
    gpu_flags_key = "khronos_validation.gpu_validation";
    gpu_flags |= GetLayerOptionFlags(gpu_flags_key, gpu_flags_option_definitions, 0);
    if (gpu_flags & CORE_VALIDATION_GPU_VALIDATION_ALL_BIT) {
        instance_state->enabled.gpu_validation = true;
    }
    if (gpu_flags & CORE_VALIDATION_GPU_VALIDATION_RESERVE_BINDING_SLOT_BIT) {
        instance_state->enabled.gpu_validation_reserve_binding_slot = true;
    }
}

void CoreChecks::PostCallRecordCreateInstance(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                                              VkInstance *pInstance, VkResult result) {
    if (VK_SUCCESS != result) return;
    InitGpuValidation();
}

void CoreChecks::PreCallRecordCreateDevice(VkPhysicalDevice gpu, const VkDeviceCreateInfo *pCreateInfo,
                                           const VkAllocationCallbacks *pAllocator, VkDevice *pDevice,
                                           std::unique_ptr<safe_VkDeviceCreateInfo> &modified_create_info) {
    // GPU Validation can possibly turn on device features, so give it a chance to change the create info.
    if (enabled.gpu_validation) {
        VkPhysicalDeviceFeatures supported_features;
        DispatchGetPhysicalDeviceFeatures(gpu, &supported_features);
        GpuPreCallRecordCreateDevice(gpu, modified_create_info, &supported_features);
    }
}

void CoreChecks::PostCallRecordCreateDevice(VkPhysicalDevice gpu, const VkDeviceCreateInfo *pCreateInfo,
                                            const VkAllocationCallbacks *pAllocator, VkDevice *pDevice, VkResult result) {
    if (VK_SUCCESS != result) return;

    ValidationObject *device_object = GetLayerDataPtr(get_dispatch_key(*pDevice), layer_data_map);
    ValidationObject *validation_data = GetValidationObject(device_object->object_dispatch, LayerObjectTypeCoreValidation);
    CoreChecks *core_checks = static_cast<CoreChecks *>(validation_data);

    // Make sure that queue_family_properties are obtained for this device's physical_device, even if the app has not
    // previously set them through an explicit API call.
    uint32_t count;
    auto pd_state = GetPhysicalDeviceState(gpu);
    DispatchGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
    pd_state->queue_family_count = count;
    pd_state->queue_family_properties.resize(std::max(static_cast<uint32_t>(pd_state->queue_family_properties.size()), count));
    DispatchGetPhysicalDeviceQueueFamilyProperties(gpu, &count, &pd_state->queue_family_properties[0]);
    // Save local link to this device's physical device state
    core_checks->physical_device_state = pd_state;
    DispatchGetPhysicalDeviceProperties(gpu, &core_checks->phys_dev_props);

    const auto &dev_ext = core_checks->device_extensions;
    auto *phys_dev_props = &core_checks->phys_dev_ext_props;

    if (dev_ext.vk_khr_push_descriptor) {
        // Get the needed push_descriptor limits
        VkPhysicalDevicePushDescriptorPropertiesKHR push_descriptor_prop;
        GetPhysicalDeviceExtProperties(gpu, dev_ext.vk_khr_push_descriptor, &push_descriptor_prop);
        phys_dev_props->max_push_descriptors = push_descriptor_prop.maxPushDescriptors;
    }

    if (enabled.gpu_validation) {
        core_checks->GpuPostCallRecordCreateDevice(&enabled);
    }

    // Store queue family data
    if ((pCreateInfo != nullptr) && (pCreateInfo->pQueueCreateInfos != nullptr)) {
        for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; ++i) {
            core_checks->queue_family_index_map.insert(
                std::make_pair(pCreateInfo->pQueueCreateInfos[i].queueFamilyIndex, pCreateInfo->pQueueCreateInfos[i].queueCount));
        }
    }
}

void CoreChecks::PreCallRecordDestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
    if (!device) return;
    if (enabled.gpu_validation) {
        GpuPreCallRecordDestroyDevice();
    }
    pipelineMap.clear();
    commandBufferMap.clear();
    renderPassMap.clear();
    // This will also delete all sets in the pool & remove them from setMap
    // All sets should be removed
    queueMap.clear();
    layer_debug_utils_destroy_device(device);
}

void CoreChecks::PreCallRecordDestroyRenderPass(VkDevice device, VkRenderPass renderPass, const VkAllocationCallbacks *pAllocator) {
    if (!renderPass) return;
    RENDER_PASS_STATE *rp_state = GetRenderPassState(renderPass);
    VK_OBJECT obj_struct = {HandleToUint64(renderPass), kVulkanObjectTypeRenderPass};
    InvalidateCommandBuffers(rp_state->cb_bindings, obj_struct);
    renderPassMap.erase(renderPass);
}

void CoreChecks::RecordCreateRenderPassState(RenderPassCreateVersion rp_version, std::shared_ptr<RENDER_PASS_STATE> &render_pass,
                                             VkRenderPass *pRenderPass) {
    render_pass->renderPass = *pRenderPass;
    auto create_info = render_pass->createInfo.ptr();

    // RecordRenderPassDAG(RENDER_PASS_VERSION_1, create_info, render_pass.get());

    for (uint32_t i = 0; i < create_info->subpassCount; ++i) {
        const VkSubpassDescription2KHR &subpass = create_info->pSubpasses[i];
        for (uint32_t j = 0; j < subpass.colorAttachmentCount; ++j) {
            // MarkAttachmentFirstUse(render_pass.get(), subpass.pColorAttachments[j].attachment, false);

            // resolve attachments are considered to be written
            if (subpass.pResolveAttachments) {
                // MarkAttachmentFirstUse(render_pass.get(), subpass.pResolveAttachments[j].attachment, false);
            }
        }
        if (subpass.pDepthStencilAttachment) {
            // MarkAttachmentFirstUse(render_pass.get(), subpass.pDepthStencilAttachment->attachment, false);
        }
        for (uint32_t j = 0; j < subpass.inputAttachmentCount; ++j) {
            // MarkAttachmentFirstUse(render_pass.get(), subpass.pInputAttachments[j].attachment, true);
        }
    }

    // Even though render_pass is an rvalue-ref parameter, still must move s.t. move assignment is invoked.
    renderPassMap[*pRenderPass] = std::move(render_pass);
}

void CoreChecks::PostCallRecordCreateRenderPass(VkDevice device, const VkRenderPassCreateInfo *pCreateInfo,
                                                const VkAllocationCallbacks *pAllocator, VkRenderPass *pRenderPass,
                                                VkResult result) {
    if (VK_SUCCESS != result) return;
    auto render_pass_state = std::make_shared<RENDER_PASS_STATE>(pCreateInfo);
    RecordCreateRenderPassState(RENDER_PASS_VERSION_1, render_pass_state, pRenderPass);
}

void CoreChecks::PostCallRecordCreateRenderPass2KHR(VkDevice device, const VkRenderPassCreateInfo2KHR *pCreateInfo,
                                                    const VkAllocationCallbacks *pAllocator, VkRenderPass *pRenderPass,
                                                    VkResult result) {
    if (VK_SUCCESS != result) return;
    auto render_pass_state = std::make_shared<RENDER_PASS_STATE>(pCreateInfo);
    RecordCreateRenderPassState(RENDER_PASS_VERSION_2, render_pass_state, pRenderPass);
}

void CoreChecks::RetireWorkOnQueue(QUEUE_STATE *pQueue, uint64_t seq) {
    std::unordered_map<VkQueue, uint64_t> otherQueueSeqs;

    // Roll this queue forward, one submission at a time.
    while (pQueue->seq < seq) {
        auto &submission = pQueue->submissions.front();

        pQueue->submissions.pop_front();
        pQueue->seq++;
    }

    // Roll other queues forward to the highest seq we saw a wait for
    for (auto qs : otherQueueSeqs) {
        RetireWorkOnQueue(GetQueueState(qs.first), qs.second);
    }
}

void CoreChecks::PostCallRecordQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo *pSubmits, VkFence fence,
                                           VkResult result) {
    uint64_t early_retire_seq = 0;
    auto pQueue = GetQueueState(queue);

    if (early_retire_seq) {
        RetireWorkOnQueue(pQueue, early_retire_seq);
    }

    if (enabled.gpu_validation) {
        GpuPostCallQueueSubmit(queue, submitCount, pSubmits, fence);
    }
}

void CoreChecks::PreCallRecordQueueSubmit(VkQueue queue, uint32_t submitCount, const VkSubmitInfo *pSubmits, VkFence fence) {
    if (enabled.gpu_validation && device_extensions.vk_ext_descriptor_indexing) {
        GpuPreCallRecordQueueSubmit(queue, submitCount, pSubmits, fence);
    }
}

void CoreChecks::RecordGetDeviceQueueState(uint32_t queue_family_index, VkQueue queue) {
    // Add queue to tracking set only if it is new
    auto queue_is_new = queues.emplace(queue);
    if (queue_is_new.second == true) {
        QUEUE_STATE *queue_state = &queueMap[queue];
        queue_state->queue = queue;
        queue_state->queueFamilyIndex = queue_family_index;
        queue_state->seq = 0;
    }
}

void CoreChecks::PostCallRecordGetDeviceQueue(VkDevice device, uint32_t queueFamilyIndex, uint32_t queueIndex, VkQueue *pQueue) {
    RecordGetDeviceQueueState(queueFamilyIndex, *pQueue);
}

void CoreChecks::PostCallRecordGetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2 *pQueueInfo, VkQueue *pQueue) {
    RecordGetDeviceQueueState(pQueueInfo->queueFamilyIndex, *pQueue);
}

void CoreChecks::PostCallRecordQueueWaitIdle(VkQueue queue, VkResult result) {
    if (VK_SUCCESS != result) return;
    QUEUE_STATE *queue_state = GetQueueState(queue);
    RetireWorkOnQueue(queue_state, queue_state->seq + queue_state->submissions.size());
}

void CoreChecks::PostCallRecordDeviceWaitIdle(VkDevice device, VkResult result) {
    if (VK_SUCCESS != result) return;
    for (auto &queue : queueMap) {
        RetireWorkOnQueue(&queue.second, queue.second.seq + queue.second.submissions.size());
    }
}

void CoreChecks::PreCallRecordDestroyShaderModule(VkDevice device, VkShaderModule shaderModule,
                                                  const VkAllocationCallbacks *pAllocator) {
    if (!shaderModule) return;
    shaderModuleMap.erase(shaderModule);
}

void CoreChecks::PreCallRecordDestroyPipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks *pAllocator) {
    if (!pipeline) return;
    PIPELINE_STATE *pipeline_state = GetPipelineState(pipeline);
    VK_OBJECT obj_struct = {HandleToUint64(pipeline), kVulkanObjectTypePipeline};
    // Any bound cmd buffers are now invalid
    InvalidateCommandBuffers(pipeline_state->cb_bindings, obj_struct);
    if (enabled.gpu_validation) {
        GpuPreCallRecordDestroyPipeline(pipeline);
    }
    pipelineMap.erase(pipeline);
}

void CoreChecks::PreCallRecordDestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout descriptorSetLayout,
                                                         const VkAllocationCallbacks *pAllocator) {
    if (!descriptorSetLayout) return;
    auto layout_it = descriptorSetLayoutMap.find(descriptorSetLayout);
    if (layout_it != descriptorSetLayoutMap.end()) {
        layout_it->second.get()->MarkDestroyed();
        descriptorSetLayoutMap.erase(layout_it);
    }
}

void CoreChecks::PreCallRecordDestroyDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                                                    const VkAllocationCallbacks *pAllocator) {
    if (!descriptorPool) return;
    DESCRIPTOR_POOL_STATE *desc_pool_state = GetDescriptorPoolState(descriptorPool);
    VK_OBJECT obj_struct = {HandleToUint64(descriptorPool), kVulkanObjectTypeDescriptorPool};
    if (desc_pool_state) {
        // Any bound cmd buffers are now invalid
        InvalidateCommandBuffers(desc_pool_state->cb_bindings, obj_struct);
        // Free sets that were in this pool
        for (auto ds : desc_pool_state->sets) {
            FreeDescriptorSet(ds);
        }
        descriptorPoolMap.erase(descriptorPool);
    }
}

// Free all command buffers in given list, removing all references/links to them using ResetCommandBufferState
void CoreChecks::FreeCommandBufferStates(COMMAND_POOL_STATE *pool_state, const uint32_t command_buffer_count,
                                         const VkCommandBuffer *command_buffers) {
    for (uint32_t i = 0; i < command_buffer_count; i++) {
        auto cb_state = GetCBState(command_buffers[i]);
        // Remove references to command buffer's state and delete
        if (cb_state) {
            // reset prior to delete, removing various references to it.
            // TODO: fix this, it's insane.
            ResetCommandBufferState(cb_state->commandBuffer);
            // Remove CBState from CB map
            commandBufferMap.erase(cb_state->commandBuffer);
            // Remove the cb debug labels
            EraseCmdDebugUtilsLabel(report_data, cb_state->commandBuffer);
            // Remove CBState from CB map
            commandBufferMap.erase(cb_state->commandBuffer);
        }
    }
}

void CoreChecks::PreCallRecordGetPhysicalDeviceProperties(VkPhysicalDevice physicalDevice,
                                                          VkPhysicalDeviceProperties *pPhysicalDeviceProperties) {
    if (enabled.gpu_validation && enabled.gpu_validation_reserve_binding_slot) {
        if (pPhysicalDeviceProperties->limits.maxBoundDescriptorSets > 1) {
            pPhysicalDeviceProperties->limits.maxBoundDescriptorSets -= 1;
        } else {
            log_msg(report_data, VK_DEBUG_REPORT_ERROR_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_PHYSICAL_DEVICE_EXT,
                    HandleToUint64(physicalDevice), "UNASSIGNED-GPU-Assisted Validation Setup Error.",
                    "Unable to reserve descriptor binding slot on a device with only one slot.");
        }
    }
}

void CoreChecks::PostCallRecordEnumeratePhysicalDevices(VkInstance instance, uint32_t *pPhysicalDeviceCount,
                                                        VkPhysicalDevice *pPhysicalDevices, VkResult result) {
    if ((NULL != pPhysicalDevices) && ((result == VK_SUCCESS || result == VK_INCOMPLETE))) {
        for (uint32_t i = 0; i < *pPhysicalDeviceCount; i++) {
            auto &phys_device_state = physical_device_map[pPhysicalDevices[i]];
            phys_device_state.phys_device = pPhysicalDevices[i];
            // Init actual features for each physical device
            DispatchGetPhysicalDeviceFeatures(pPhysicalDevices[i], &phys_device_state.features2.features);
        }
    }
}

void CoreChecks::PreCallRecordFreeCommandBuffers(VkDevice device, VkCommandPool commandPool, uint32_t commandBufferCount,
                                                 const VkCommandBuffer *pCommandBuffers) {
    FreeCommandBufferStates(nullptr, commandBufferCount, pCommandBuffers);
}

// For given cb_nodes, invalidate them and track object causing invalidation
void CoreChecks::InvalidateCommandBuffers(std::unordered_set<CMD_BUFFER_STATE *> const &cb_nodes, VK_OBJECT obj) {
    for (auto cb_node : cb_nodes) {
        if (cb_node->state == CB_RECORDING) {
            log_msg(report_data, VK_DEBUG_REPORT_WARNING_BIT_EXT, VK_DEBUG_REPORT_OBJECT_TYPE_COMMAND_BUFFER_EXT,
                    HandleToUint64(cb_node->commandBuffer), kVUID_Core_DrawState_InvalidCommandBuffer,
                    "Invalidating a command buffer that's currently being recorded: %s.",
                    report_data->FormatHandle(cb_node->commandBuffer).c_str());
            cb_node->state = CB_INVALID_INCOMPLETE;
        } else if (cb_node->state == CB_RECORDED) {
            cb_node->state = CB_INVALID_COMPLETE;
        }
        cb_node->broken_bindings.push_back(obj);

        // if secondary, then propagate the invalidation to the primaries that will call us.
        if (cb_node->createInfo.level == VK_COMMAND_BUFFER_LEVEL_SECONDARY) {
            InvalidateCommandBuffers(cb_node->linkedCommandBuffers, obj);
        }
    }
}

void CoreChecks::UpdateDrawState(CMD_BUFFER_STATE *cb_state, const VkPipelineBindPoint bind_point) {
    auto const &state = cb_state->lastBound[bind_point];
    PIPELINE_STATE *pPipe = state.pipeline_state;
    if (VK_NULL_HANDLE != state.pipeline_layout) {
        for (const auto &set_binding_pair : pPipe->active_slots) {
            uint32_t setIndex = set_binding_pair.first;
            // Pull the set node
            cvdescriptorset::DescriptorSet *descriptor_set = state.boundDescriptorSets[setIndex];
            if (!descriptor_set->IsPushDescriptor()) {
                // For the "bindless" style resource usage with many descriptors, need to optimize command <-> descriptor binding
                const cvdescriptorset::PrefilterBindRequestMap reduced_map(*descriptor_set, set_binding_pair.second, cb_state);
                const auto &binding_req_map = reduced_map.Map();

                // Bind this set and its active descriptor resources to the command buffer
                descriptor_set->UpdateDrawState(this, cb_state, binding_req_map);
                // For given active slots record updated images & buffers
            }
        }
    }
    if (!pPipe->vertex_binding_descriptions_.empty()) {
        cb_state->vertex_buffer_used = true;
    }
}
RENDER_PASS_STATE *CoreChecks::GetRenderPassState(VkRenderPass renderpass) {
    auto it = renderPassMap.find(renderpass);
    if (it == renderPassMap.end()) {
        return nullptr;
    }
    return it->second.get();
}

std::shared_ptr<RENDER_PASS_STATE> CoreChecks::GetRenderPassStateSharedPtr(VkRenderPass renderpass) {
    auto it = renderPassMap.find(renderpass);
    if (it == renderPassMap.end()) {
        return nullptr;
    }
    return it->second;
}

bool CoreChecks::PreCallValidateCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                                        const VkGraphicsPipelineCreateInfo *pCreateInfos,
                                                        const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines,
                                                        void *cgpl_state_data) {
    bool skip = false;
    create_graphics_pipeline_api_state *cgpl_state = reinterpret_cast<create_graphics_pipeline_api_state *>(cgpl_state_data);
    cgpl_state->pipe_state.reserve(count);
    for (uint32_t i = 0; i < count; i++) {
        cgpl_state->pipe_state.push_back(std::unique_ptr<PIPELINE_STATE>(new PIPELINE_STATE));
        (cgpl_state->pipe_state)[i]->initGraphicsPipeline(&pCreateInfos[i],
                                                          GetRenderPassStateSharedPtr(pCreateInfos[i].renderPass));
        (cgpl_state->pipe_state)[i]->pipeline_layout = *GetPipelineLayout(pCreateInfos[i].layout);
    }

    return skip;
}

// GPU validation may replace pCreateInfos for the down-chain call
void CoreChecks::PreCallRecordCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                                      const VkGraphicsPipelineCreateInfo *pCreateInfos,
                                                      const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines,
                                                      void *cgpl_state_data) {
    create_graphics_pipeline_api_state *cgpl_state = reinterpret_cast<create_graphics_pipeline_api_state *>(cgpl_state_data);
    cgpl_state->pCreateInfos = pCreateInfos;
    // GPU Validation may replace instrumented shaders with non-instrumented ones, so allow it to modify the createinfos.
    if (enabled.gpu_validation) {
        cgpl_state->gpu_create_infos = GpuPreCallRecordCreateGraphicsPipelines(pipelineCache, count, pCreateInfos, pAllocator,
                                                                               pPipelines, cgpl_state->pipe_state);
        cgpl_state->pCreateInfos = reinterpret_cast<VkGraphicsPipelineCreateInfo *>(cgpl_state->gpu_create_infos.data());
    }
}

void CoreChecks::PostCallRecordCreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                                       const VkGraphicsPipelineCreateInfo *pCreateInfos,
                                                       const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines,
                                                       VkResult result, void *cgpl_state_data) {
    create_graphics_pipeline_api_state *cgpl_state = reinterpret_cast<create_graphics_pipeline_api_state *>(cgpl_state_data);
    // This API may create pipelines regardless of the return value
    for (uint32_t i = 0; i < count; i++) {
        if (pPipelines[i] != VK_NULL_HANDLE) {
            (cgpl_state->pipe_state)[i]->pipeline = pPipelines[i];
            pipelineMap[pPipelines[i]] = std::move((cgpl_state->pipe_state)[i]);
        }
    }
    // GPU val needs clean up regardless of result
    if (enabled.gpu_validation) {
        GpuPostCallRecordCreateGraphicsPipelines(count, pCreateInfos, pAllocator, pPipelines);
        cgpl_state->gpu_create_infos.clear();
    }
    cgpl_state->pipe_state.clear();
}

void CoreChecks::PostCallRecordCreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                                      const VkComputePipelineCreateInfo *pCreateInfos,
                                                      const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines,
                                                      VkResult result, void *pipe_state_data) {
    std::vector<std::unique_ptr<PIPELINE_STATE>> *pipe_state =
        reinterpret_cast<std::vector<std::unique_ptr<PIPELINE_STATE>> *>(pipe_state_data);

    // This API may create pipelines regardless of the return value
    for (uint32_t i = 0; i < count; i++) {
        if (pPipelines[i] != VK_NULL_HANDLE) {
            (*pipe_state)[i]->pipeline = pPipelines[i];
            pipelineMap[pPipelines[i]] = std::move((*pipe_state)[i]);
        }
    }
}

void CoreChecks::PostCallRecordCreateRayTracingPipelinesNV(VkDevice device, VkPipelineCache pipelineCache, uint32_t count,
                                                           const VkRayTracingPipelineCreateInfoNV *pCreateInfos,
                                                           const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines,
                                                           VkResult result, void *pipe_state_data) {
    vector<std::unique_ptr<PIPELINE_STATE>> *pipe_state =
        reinterpret_cast<vector<std::unique_ptr<PIPELINE_STATE>> *>(pipe_state_data);
    // This API may create pipelines regardless of the return value
    for (uint32_t i = 0; i < count; i++) {
        if (pPipelines[i] != VK_NULL_HANDLE) {
            (*pipe_state)[i]->pipeline = pPipelines[i];
            pipelineMap[pPipelines[i]] = std::move((*pipe_state)[i]);
        }
    }
}

void CoreChecks::PostCallRecordCreateDescriptorSetLayout(VkDevice device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                                         const VkAllocationCallbacks *pAllocator, VkDescriptorSetLayout *pSetLayout,
                                                         VkResult result) {
    if (VK_SUCCESS != result) return;
    descriptorSetLayoutMap[*pSetLayout] = std::make_shared<cvdescriptorset::DescriptorSetLayout>(pCreateInfo, *pSetLayout);
}

enum DSL_DESCRIPTOR_GROUPS {
    DSL_TYPE_SAMPLERS = 0,
    DSL_TYPE_UNIFORM_BUFFERS,
    DSL_TYPE_STORAGE_BUFFERS,
    DSL_TYPE_SAMPLED_IMAGES,
    DSL_TYPE_STORAGE_IMAGES,
    DSL_TYPE_INPUT_ATTACHMENTS,
    DSL_TYPE_INLINE_UNIFORM_BLOCK,
    DSL_NUM_DESCRIPTOR_GROUPS
};

void CoreChecks::PreCallRecordCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo *pCreateInfo,
                                                   const VkAllocationCallbacks *pAllocator, VkPipelineLayout *pPipelineLayout,
                                                   void *cpl_state_data) {
    create_pipeline_layout_api_state *cpl_state = reinterpret_cast<create_pipeline_layout_api_state *>(cpl_state_data);
    if (enabled.gpu_validation) {
        GpuPreCallCreatePipelineLayout(pCreateInfo, pAllocator, pPipelineLayout, &cpl_state->new_layouts,
                                       &cpl_state->modified_create_info);
    }
}

void CoreChecks::PostCallRecordCreatePipelineLayout(VkDevice device, const VkPipelineLayoutCreateInfo *pCreateInfo,
                                                    const VkAllocationCallbacks *pAllocator, VkPipelineLayout *pPipelineLayout,
                                                    VkResult result) {
    // Clean up GPU validation
    if (enabled.gpu_validation) {
        GpuPostCallCreatePipelineLayout(result);
    }
    if (VK_SUCCESS != result) return;
    std::unique_ptr<PIPELINE_LAYOUT_STATE> pipeline_layout_state(new PIPELINE_LAYOUT_STATE{});
    pipeline_layout_state->layout = *pPipelineLayout;
    pipeline_layout_state->set_layouts.resize(pCreateInfo->setLayoutCount);
    pipelineLayoutMap[*pPipelineLayout] = std::move(pipeline_layout_state);
}

// Ensure the pool contains enough descriptors and descriptor sets to satisfy
// an allocation request. Fills common_data with the total number of descriptors of each type required,
// as well as DescriptorSetLayout ptrs used for later update.
bool CoreChecks::PreCallValidateAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo *pAllocateInfo,
                                                       VkDescriptorSet *pDescriptorSets, void *ads_state_data) {
    // Always update common data
    cvdescriptorset::AllocateDescriptorSetsData *ads_state =
        reinterpret_cast<cvdescriptorset::AllocateDescriptorSetsData *>(ads_state_data);
    UpdateAllocateDescriptorSetsData(pAllocateInfo, ads_state);
    // All state checks for AllocateDescriptorSets is done in single function
    return VK_SUCCESS;
}

// Allocation state was good and call down chain was made so update state based on allocating descriptor sets
void CoreChecks::PostCallRecordAllocateDescriptorSets(VkDevice device, const VkDescriptorSetAllocateInfo *pAllocateInfo,
                                                      VkDescriptorSet *pDescriptorSets, VkResult result, void *ads_state_data) {
    if (VK_SUCCESS != result) return;
    // All the updates are contained in a single cvdescriptorset function
    cvdescriptorset::AllocateDescriptorSetsData *ads_state =
        reinterpret_cast<cvdescriptorset::AllocateDescriptorSetsData *>(ads_state_data);
    PerformAllocateDescriptorSets(pAllocateInfo, pDescriptorSets, ads_state);
}

void CoreChecks::PreCallRecordUpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                                                   const VkWriteDescriptorSet *pDescriptorWrites, uint32_t descriptorCopyCount,
                                                   const VkCopyDescriptorSet *pDescriptorCopies) {
    cvdescriptorset::PerformUpdateDescriptorSets(this, descriptorWriteCount, pDescriptorWrites, descriptorCopyCount,
                                                 pDescriptorCopies);
}

void CoreChecks::PostCallRecordAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo *pCreateInfo,
                                                      VkCommandBuffer *pCommandBuffer, VkResult result) {
    if (VK_SUCCESS != result) return;
    for (uint32_t i = 0; i < pCreateInfo->commandBufferCount; i++) {
        // Add command buffer to its commandPool map
        std::unique_ptr<CMD_BUFFER_STATE> pCB(new CMD_BUFFER_STATE{});
        pCB->createInfo = *pCreateInfo;
        pCB->device = device;
        // Add command buffer to map
        commandBufferMap[pCommandBuffer[i]] = std::move(pCB);
        ResetCommandBufferState(pCommandBuffer[i]);
    }
}

void CoreChecks::PostCallRecordResetCommandBuffer(VkCommandBuffer commandBuffer, VkCommandBufferResetFlags flags, VkResult result) {
    if (VK_SUCCESS == result) {
        ResetCommandBufferState(commandBuffer);
    }
}

static const char *GetPipelineTypeName(VkPipelineBindPoint pipelineBindPoint) {
    switch (pipelineBindPoint) {
        case VK_PIPELINE_BIND_POINT_GRAPHICS:
            return "graphics";
        case VK_PIPELINE_BIND_POINT_COMPUTE:
            return "compute";
        case VK_PIPELINE_BIND_POINT_RAY_TRACING_NV:
            return "ray-tracing";
        default:
            return "unknown";
    }
}

void CoreChecks::PreCallRecordCmdBindPipeline(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                              VkPipeline pipeline) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    assert(cb_state);

    auto pipe_state = GetPipelineState(pipeline);
    cb_state->lastBound[pipelineBindPoint].pipeline_state = pipe_state;
    AddCommandBufferBinding(&pipe_state->cb_bindings, {HandleToUint64(pipeline), kVulkanObjectTypePipeline}, cb_state);
}

// Update pipeline_layout bind points applying the "Pipeline Layout Compatibility" rules
void CoreChecks::UpdateLastBoundDescriptorSets(CMD_BUFFER_STATE *cb_state, VkPipelineBindPoint pipeline_bind_point,
                                               const PIPELINE_LAYOUT_STATE *pipeline_layout, uint32_t first_set, uint32_t set_count,
                                               const std::vector<cvdescriptorset::DescriptorSet *> descriptor_sets,
                                               uint32_t dynamic_offset_count, const uint32_t *p_dynamic_offsets) {
    // Defensive
    assert(set_count);
    if (0 == set_count) return;
    assert(pipeline_layout);
    if (!pipeline_layout) return;

    uint32_t required_size = first_set + set_count;
    const uint32_t last_binding_index = required_size - 1;

    // Some useful shorthand
    auto &last_bound = cb_state->lastBound[pipeline_bind_point];

    auto &bound_sets = last_bound.boundDescriptorSets;
    auto &dynamic_offsets = last_bound.dynamicOffsets;
    auto &bound_compat_ids = last_bound.compat_id_for_set;
    auto &pipe_compat_ids = pipeline_layout->compat_for_set;

    const uint32_t current_size = static_cast<uint32_t>(bound_sets.size());
    assert(current_size == dynamic_offsets.size());
    assert(current_size == bound_compat_ids.size());

    // We need this three times in this function, but nowhere else
    auto push_descriptor_cleanup = [&last_bound](const cvdescriptorset::DescriptorSet *ds) -> bool {
        if (ds && ds->IsPushDescriptor()) {
            assert(ds == last_bound.push_descriptor_set.get());
            last_bound.push_descriptor_set = nullptr;
            return true;
        }
        return false;
    };

    // Clean up the "disturbed" before and after the range to be set
    if (required_size < current_size) {
        if (bound_compat_ids[last_binding_index] != pipe_compat_ids[last_binding_index]) {
            // We're disturbing those after last, we'll shrink below, but first need to check for and cleanup the push_descriptor
            for (auto set_idx = required_size; set_idx < current_size; ++set_idx) {
                if (push_descriptor_cleanup(bound_sets[set_idx])) break;
            }
        } else {
            // We're not disturbing past last, so leave the upper binding data alone.
            required_size = current_size;
        }
    }

    // We resize if we need more set entries or if those past "last" are disturbed
    if (required_size != current_size) {
        // TODO: put these size tied things in a struct (touches many lines)
        bound_sets.resize(required_size);
        dynamic_offsets.resize(required_size);
        bound_compat_ids.resize(required_size);
    }

    // For any previously bound sets, need to set them to "invalid" if they were disturbed by this update
    for (uint32_t set_idx = 0; set_idx < first_set; ++set_idx) {
        if (bound_compat_ids[set_idx] != pipe_compat_ids[set_idx]) {
            push_descriptor_cleanup(bound_sets[set_idx]);
            bound_sets[set_idx] = nullptr;
            dynamic_offsets[set_idx].clear();
            bound_compat_ids[set_idx] = pipe_compat_ids[set_idx];
        }
    }

    // Now update the bound sets with the input sets
    const uint32_t *input_dynamic_offsets = p_dynamic_offsets;  // "read" pointer for dynamic offset data
    for (uint32_t input_idx = 0; input_idx < set_count; input_idx++) {
        auto set_idx = input_idx + first_set;  // set_idx is index within layout, input_idx is index within input descriptor sets
        cvdescriptorset::DescriptorSet *descriptor_set = descriptor_sets[input_idx];

        // Record binding (or push)
        if (descriptor_set != last_bound.push_descriptor_set.get()) {
            // Only cleanup the push descriptors if they aren't the currently used set.
            push_descriptor_cleanup(bound_sets[set_idx]);
        }
        bound_sets[set_idx] = descriptor_set;

        if (descriptor_set) {
            auto set_dynamic_descriptor_count = descriptor_set->GetDynamicDescriptorCount();
            // TODO: Add logic for tracking push_descriptor offsets (here or in caller)
            if (set_dynamic_descriptor_count && input_dynamic_offsets) {
                const uint32_t *end_offset = input_dynamic_offsets + set_dynamic_descriptor_count;
                dynamic_offsets[set_idx] = std::vector<uint32_t>(input_dynamic_offsets, end_offset);
                input_dynamic_offsets = end_offset;
                assert(input_dynamic_offsets <= (p_dynamic_offsets + dynamic_offset_count));
            } else {
                dynamic_offsets[set_idx].clear();
            }
            if (!descriptor_set->IsPushDescriptor()) {
                // Can't cache validation of push_descriptors
                cb_state->validated_descriptor_sets.insert(descriptor_set);
            }
        }
    }
}

// Update the bound state for the bind point, including the effects of incompatible pipeline layouts
void CoreChecks::PreCallRecordCmdBindDescriptorSets(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                                    VkPipelineLayout layout, uint32_t firstSet, uint32_t setCount,
                                                    const VkDescriptorSet *pDescriptorSets, uint32_t dynamicOffsetCount,
                                                    const uint32_t *pDynamicOffsets) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    auto pipeline_layout = GetPipelineLayout(layout);
    std::vector<cvdescriptorset::DescriptorSet *> descriptor_sets;
    descriptor_sets.reserve(setCount);

    // Construct a list of the descriptors
    bool found_non_null = false;
    for (uint32_t i = 0; i < setCount; i++) {
        cvdescriptorset::DescriptorSet *descriptor_set = GetSetNode(pDescriptorSets[i]);
        descriptor_sets.emplace_back(descriptor_set);
        found_non_null |= descriptor_set != nullptr;
    }
    if (found_non_null) {  // which implies setCount > 0
        UpdateLastBoundDescriptorSets(cb_state, pipelineBindPoint, pipeline_layout, firstSet, setCount, descriptor_sets,
                                      dynamicOffsetCount, pDynamicOffsets);
        cb_state->lastBound[pipelineBindPoint].pipeline_layout = layout;
    }
}

void CoreChecks::RecordCmdPushDescriptorSetState(CMD_BUFFER_STATE *cb_state, VkPipelineBindPoint pipelineBindPoint,
                                                 VkPipelineLayout layout, uint32_t set, uint32_t descriptorWriteCount,
                                                 const VkWriteDescriptorSet *pDescriptorWrites) {
    const auto &pipeline_layout = GetPipelineLayout(layout);
    // Short circuit invalid updates
    if (!pipeline_layout || (set >= pipeline_layout->set_layouts.size()) || !pipeline_layout->set_layouts[set] ||
        !pipeline_layout->set_layouts[set]->IsPushDescriptor())
        return;

    // We need a descriptor set to update the bindings with, compatible with the passed layout
    const auto dsl = pipeline_layout->set_layouts[set];
    auto &last_bound = cb_state->lastBound[pipelineBindPoint];
    auto &push_descriptor_set = last_bound.push_descriptor_set;
    // If we are disturbing the current push_desriptor_set clear it
    if (!push_descriptor_set || !CompatForSet(set, last_bound.compat_id_for_set, pipeline_layout->compat_for_set)) {
        push_descriptor_set.reset(new cvdescriptorset::DescriptorSet(0, 0, dsl, 0, this));
    }

    std::vector<cvdescriptorset::DescriptorSet *> descriptor_sets = {push_descriptor_set.get()};
    UpdateLastBoundDescriptorSets(cb_state, pipelineBindPoint, pipeline_layout, set, 1, descriptor_sets, 0, nullptr);
    last_bound.pipeline_layout = layout;

    // Now that we have either the new or extant push_descriptor set ... do the write updates against it
    push_descriptor_set->PerformPushDescriptorsUpdate(descriptorWriteCount, pDescriptorWrites);
}

void CoreChecks::PreCallRecordCmdPushDescriptorSetKHR(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                                      VkPipelineLayout layout, uint32_t set, uint32_t descriptorWriteCount,
                                                      const VkWriteDescriptorSet *pDescriptorWrites) {
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    RecordCmdPushDescriptorSetState(cb_state, pipelineBindPoint, layout, set, descriptorWriteCount, pDescriptorWrites);
}

static VkDeviceSize GetIndexAlignment(VkIndexType indexType) {
    switch (indexType) {
        case VK_INDEX_TYPE_UINT16:
            return 2;
        case VK_INDEX_TYPE_UINT32:
            return 4;
        default:
            // Not a real index type. Express no alignment requirement here; we expect upper layer
            // to have already picked up on the enum being nonsense.
            return 1;
    }
}

void CoreChecks::PreCallRecordCmdWaitEvents(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent *pEvents,
                                            VkPipelineStageFlags sourceStageMask, VkPipelineStageFlags dstStageMask,
                                            uint32_t memoryBarrierCount, const VkMemoryBarrier *pMemoryBarriers,
                                            uint32_t bufferMemoryBarrierCount, const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                                            uint32_t imageMemoryBarrierCount, const VkImageMemoryBarrier *pImageMemoryBarriers) {
    if (enabled.gpu_validation) {
        GpuPreCallValidateCmdWaitEvents(sourceStageMask);
    }
}

void CoreChecks::PreCallRecordCmdBeginDebugUtilsLabelEXT(VkCommandBuffer commandBuffer, const VkDebugUtilsLabelEXT *pLabelInfo) {
    BeginCmdDebugUtilsLabel(report_data, commandBuffer, pLabelInfo);
}

void CoreChecks::PostCallRecordCmdEndDebugUtilsLabelEXT(VkCommandBuffer commandBuffer) {
    EndCmdDebugUtilsLabel(report_data, commandBuffer);
}

void CoreChecks::PreCallRecordCmdInsertDebugUtilsLabelEXT(VkCommandBuffer commandBuffer, const VkDebugUtilsLabelEXT *pLabelInfo) {
    InsertCmdDebugUtilsLabel(report_data, commandBuffer, pLabelInfo);

    // Squirrel away an easily accessible copy.
    CMD_BUFFER_STATE *cb_state = GetCBState(commandBuffer);
    cb_state->debug_label = LoggingLabel(pLabelInfo);
}
